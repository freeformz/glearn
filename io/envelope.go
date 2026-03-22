package io

import (
	"hash/crc32"
)

// Version is the glearn library version embedded in serialized models.
const Version = "0.1.0"

// Envelope wraps a serialized model with metadata and integrity checking.
// The workflow is:
//  1. Save: serialize model to bytes (inner), compute CRC32, wrap in envelope, serialize envelope (outer)
//  2. Load: deserialize envelope, verify CRC32 of Data matches Checksum, deserialize Data into model
type Envelope struct {
	Format   string `json:"format" msgpack:"format"`     // "json", "gob", "msgpack"
	Type     string `json:"type" msgpack:"type"`          // Go type name of the model
	Version  string `json:"version" msgpack:"version"`    // glearn version
	Checksum uint32 `json:"checksum" msgpack:"checksum"`  // CRC32 of Data
	Data     []byte `json:"data" msgpack:"data"`          // serialized model bytes
}

// computeChecksum returns the CRC32 (IEEE) checksum of the given data.
func computeChecksum(data []byte) uint32 {
	return crc32.ChecksumIEEE(data)
}

// verifyChecksum returns true if the CRC32 checksum of data matches expected.
func verifyChecksum(data []byte, expected uint32) bool {
	return computeChecksum(data) == expected
}
