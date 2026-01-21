using System;
using System.IO;
using UnityEngine;

// PROTOCOL DESIGN:
// Structure: [length (4 bytes)][success (1 byte)][if success: face_id (4 bytes)][if success: seq_num (4 bytes)]
// Protocol Size = 13 bytes if success, else 5 bytes

public class IDPacket
{
    public bool success;
    public int faceId;
    public int seqNum;
    
    public IDPacket(bool success, int faceId = -1, int seqNum = -1)
    {
        this.success = success;
        this.faceId = faceId;
        this.seqNum = seqNum;
    }
    
    public byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Write success flag
            writer.Write(success);
            
            if (success)
            {
                writer.Write(faceId);
                writer.Write(seqNum);
            }
            
            byte[] packetData = ms.ToArray();
            
            // Add length prefix
            using (MemoryStream finalMs = new MemoryStream())
            using (BinaryWriter finalWriter = new BinaryWriter(finalMs))
            {
                finalWriter.Write(packetData.Length); // Length prefix (little-endian)
                finalWriter.Write(packetData);
                
                return finalMs.ToArray();
            }
        }
    }
    
    public static IDPacket Deserialize(byte[] data)
    {
        try
        {
            if (data == null || data.Length < 4)
                return null;
            
            using (MemoryStream ms = new MemoryStream(data))
            using (BinaryReader reader = new BinaryReader(ms))
            {
                // Read length prefix (little-endian)
                int totalLength = reader.ReadInt32();
                
                // Verify we have enough data
                if (data.Length < 4 + totalLength)
                {
                    Debug.LogWarning($"Not enough data: Have {data.Length}, need {4 + totalLength}");
                    return null;
                }
                
                // Read success flag
                bool successFlag = reader.ReadBoolean();
                
                if (!successFlag)
                {
                    return new IDPacket(false);
                }
                else
                {
                    int faceId = reader.ReadInt32();
                    int seqNum = reader.ReadInt32();
                    
                    return new IDPacket(true, faceId, seqNum);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"IDPacket deserialization error: {e}");
            return null;
        }
    }
    
    public static IDPacket Deserialize(byte[] data, int offset, int length)
    {
        if (data == null || offset < 0 || length <= 0 || offset + length > data.Length)
            return null;
            
        byte[] subset = new byte[length];
        Buffer.BlockCopy(data, offset, subset, 0, length);
        return Deserialize(subset);
    }
    
    public override string ToString()
    {
        return $"IDPacket(success={success}, faceId={faceId}, seqNum={seqNum})";
    }
}