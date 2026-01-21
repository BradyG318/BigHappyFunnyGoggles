using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// PROTOCOL DESIGN:
// Header [TCP Packet Length (4 bytes) | Sequence Number (4 bytes)]
// Payload [NumFaces (1 byte) | RecentIDs (5 * 4 bytes) | CropSizes (1 or 10 * 4 bytes) | Crops (1 or 10 * variable size)]
// Protocol Size = 29 bytes + (1 or 10 * (variable size + 4 bytes))

public class FacePacket : IDisposable
{
    public int seqNum;
    public List<Texture2D> faceCrops;
    public List<int?> recentIds;
    
    public FacePacket(int seqNum, List<Texture2D> faceCrops, List<int?> recentIds = null)
    {
        this.seqNum = seqNum;
        this.faceCrops = faceCrops ?? new List<Texture2D>();
        
        // Initialize with 5 null values
        this.recentIds = new List<int?> { null, null, null, null, null };
        
        if (recentIds != null)
        {
            for (int i = 0; i < Mathf.Min(5, recentIds.Count); i++)
            {
                this.recentIds[i] = recentIds[i];
            }
        }
    }
    
    public FacePacket(int seqNum, Texture2D singleFaceCrop, List<int?> recentIds = null) : 
        this(seqNum, new List<Texture2D> { singleFaceCrop }, recentIds)
    {
    }
    
    public byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            int numFaces = faceCrops.Count;
            
            // Write number of faces
            writer.Write((byte)numFaces);
            
            // Write recent IDs (always 5)
            for (int i = 0; i < 5; i++)
            {
                int idValue = recentIds[i].HasValue ? recentIds[i].Value : -1;
                writer.Write(idValue);
            }
            
            // Compress crops and record sizes
            List<byte[]> compressedCrops = new List<byte[]>();
            List<int> cropSizes = new List<int>();
            
            foreach (Texture2D faceCrop in faceCrops)
            {
                if (faceCrop == null)
                {
                    compressedCrops.Add(new byte[0]);
                    cropSizes.Add(0);
                }
                else
                {
                    // IMPORTANT: Unity encodes as RGB, Python server expects RGB and will convert to BGR
                    byte[] jpegBytes = faceCrop.EncodeToJPG(90);
                    compressedCrops.Add(jpegBytes);
                    cropSizes.Add(jpegBytes.Length);
                }
            }
            
            // Write crop sizes
            foreach (int size in cropSizes)
            {
                writer.Write(size);
            }
            
            // Write crop data
            foreach (byte[] cropData in compressedCrops)
            {
                writer.Write(cropData);
            }
            
            byte[] payload = ms.ToArray();
            
            // Add header: total length (including seqNum) + seqNum
            using (MemoryStream finalMs = new MemoryStream())
            using (BinaryWriter finalWriter = new BinaryWriter(finalMs))
            {
                // Total length = payload length + 4 bytes for seqNum
                int totalLength = payload.Length + 4;
                finalWriter.Write(totalLength);     // Packet length (little-endian)
                finalWriter.Write(seqNum);          // Sequence number (little-endian)
                finalWriter.Write(payload);         // Payload
                
                return finalMs.ToArray();
            }
        }
    }
    
    public static FacePacket Deserialize(byte[] data)
    {
        try
        {
            if (data == null || data.Length < 8) // Need at least length(4) + seqNum(4)
                return null;
            
            using (MemoryStream ms = new MemoryStream(data))
            using (BinaryReader reader = new BinaryReader(ms))
            {
                // Read total length (little-endian)
                int totalLength = reader.ReadInt32();
                
                // Verify we have enough data
                if (data.Length < 4 + totalLength)
                {
                    Debug.LogWarning($"Not enough data: Have {data.Length}, need {4 + totalLength}");
                    return null;
                }
                
                // Read sequence number (little-endian)
                int seqNum = reader.ReadInt32();
                
                // Read number of faces
                if (ms.Position >= ms.Length)
                    return null;
                    
                byte numFaces = reader.ReadByte();
                
                // Read recent IDs (always 5)
                List<int?> recentIds = new List<int?>();
                for (int i = 0; i < 5; i++)
                {
                    if (ms.Position + 4 > ms.Length)
                        return null;
                        
                    int faceId = reader.ReadInt32();
                    recentIds.Add(faceId != -1 ? (int?)faceId : null);
                }
                
                // Read crop sizes
                List<int> cropSizes = new List<int>();
                for (int i = 0; i < numFaces; i++)
                {
                    if (ms.Position + 4 > ms.Length)
                        return null;
                        
                    int cropSize = reader.ReadInt32();
                    cropSizes.Add(cropSize);
                }
                
                // Read and decode crops
                List<Texture2D> faceCrops = new List<Texture2D>();
                for (int i = 0; i < numFaces; i++)
                {
                    int cropSize = cropSizes[i];
                    if (cropSize == 0)
                    {
                        faceCrops.Add(null);
                    }
                    else
                    {
                        if (ms.Position + cropSize > ms.Length)
                            return null;
                            
                        byte[] cropData = reader.ReadBytes(cropSize);
                        Texture2D faceCrop = new Texture2D(2, 2, TextureFormat.RGB24, false);
                        
                        try
                        {
                            if (faceCrop.LoadImage(cropData, false)) // markNonReadable = false
                            {
                                // Note: If receiving from Python server, this would be BGR
                                // But since we're not receiving FacePackets in Unity, this is just for completeness
                                faceCrops.Add(faceCrop);
                            }
                            else
                            {
                                Debug.LogWarning($"Failed to decode image {i}");
                                UnityEngine.Object.Destroy(faceCrop);
                                faceCrops.Add(null);
                            }
                        }
                        catch (Exception e)
                        {
                            Debug.LogError($"Error decoding image {i}: {e}");
                            if (faceCrop != null)
                                UnityEngine.Object.Destroy(faceCrop);
                            faceCrops.Add(null);
                        }
                    }
                }
                
                return new FacePacket(seqNum, faceCrops, recentIds);
            }
        }
        catch (EndOfStreamException)
        {
            Debug.LogError("FacePacket deserialization: Unexpected end of stream");
            return null;
        }
        catch (Exception e)
        {
            Debug.LogError($"FacePacket deserialization error: {e}");
            return null;
        }
    }
    
    public static FacePacket Deserialize(byte[] data, int offset, int length)
    {
        if (data == null || offset < 0 || length <= 0 || offset + length > data.Length)
            return null;
            
        byte[] subset = new byte[length];
        Buffer.BlockCopy(data, offset, subset, 0, length);
        return Deserialize(subset);
    }
    
    // Helper method to create a Texture2D from camera frame data
    public static Texture2D CreateTextureFromCameraFrame(byte[] frameData, int width, int height)
    {
        Texture2D texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        
        // Note: Camera frames are often in YUV or other formats
        // This assumes RGB24 format from Unity's WebCamTexture
        texture.LoadRawTextureData(frameData);
        texture.Apply();
        
        return texture;
    }
    
    // Helper method to convert Texture2D to BGR byte array (for debugging)
    public static byte[] TextureToBGRBytes(Texture2D texture)
    {
        if (texture == null) return null;
        
        Color32[] pixels = texture.GetPixels32();
        byte[] bgrBytes = new byte[pixels.Length * 3];
        
        for (int i = 0; i < pixels.Length; i++)
        {
            int idx = i * 3;
            // Convert RGB to BGR
            bgrBytes[idx + 2] = pixels[i].r;     // R -> position 2
            bgrBytes[idx + 1] = pixels[i].g;     // G -> position 1  
            bgrBytes[idx] = pixels[i].b;         // B -> position 0
        }
        
        return bgrBytes;
    }
    
    // Dispose pattern to clean up textures
    private bool disposed = false;
    
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (!disposed)
        {
            if (disposing)
            {
                // Dispose managed resources
                if (faceCrops != null)
                {
                    foreach (Texture2D texture in faceCrops)
                    {
                        if (texture != null)
                        {
                            UnityEngine.Object.Destroy(texture);
                        }
                    }
                    faceCrops.Clear();
                }
            }
            
            disposed = true;
        }
    }
    
    ~FacePacket()
    {
        Dispose(false);
    }
    
    public override string ToString()
    {
        return $"FacePacket(seqNum={seqNum}, faces={faceCrops?.Count ?? 0})";
    }
}