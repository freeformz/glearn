package io

import (
	"fmt"
	"io"
	"os"
	"reflect"

	"github.com/vmihailenco/msgpack/v5"
)

// SaveMsgpack saves a model to a writer using MessagePack with CRC32 checksum.
// The model is first serialized to MessagePack bytes (inner), then wrapped in an
// Envelope and serialized again (outer).
func SaveMsgpack(w io.Writer, model any) error {
	// Serialize the model to msgpack bytes (inner payload).
	data, err := msgpack.Marshal(model)
	if err != nil {
		return fmt.Errorf("glearn/io: msgpack save failed: could not marshal model: %w", err)
	}

	// Build envelope with checksum.
	env := Envelope{
		Format:   "msgpack",
		Type:     reflect.TypeOf(model).String(),
		Version:  Version,
		Checksum: computeChecksum(data),
		Data:     data,
	}

	// Serialize the envelope to the writer.
	encoder := msgpack.NewEncoder(w)
	if err := encoder.Encode(env); err != nil {
		return fmt.Errorf("glearn/io: msgpack save failed: could not encode envelope: %w", err)
	}

	return nil
}

// LoadMsgpack loads a model from MessagePack format, verifying the CRC32 checksum.
// The model parameter must be a pointer to the target type.
func LoadMsgpack(r io.Reader, model any) error {
	// Decode the envelope.
	var env Envelope
	decoder := msgpack.NewDecoder(r)
	if err := decoder.Decode(&env); err != nil {
		return fmt.Errorf("glearn/io: msgpack load failed: could not decode envelope: %w", err)
	}

	// Verify checksum.
	if !verifyChecksum(env.Data, env.Checksum) {
		return fmt.Errorf("glearn/io: msgpack load failed: checksum mismatch")
	}

	// Deserialize the model from the inner data.
	if err := msgpack.Unmarshal(env.Data, model); err != nil {
		return fmt.Errorf("glearn/io: msgpack load failed: could not unmarshal model: %w", err)
	}

	return nil
}

// SaveMsgpackFile saves a model to a file using MessagePack encoding.
func SaveMsgpackFile(path string, model any) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("glearn/io: msgpack save to file failed: %w", err)
	}
	defer f.Close()

	if err := SaveMsgpack(f, model); err != nil {
		return err
	}

	return f.Close()
}

// LoadMsgpackFile loads a model from a MessagePack-encoded file.
func LoadMsgpackFile(path string, model any) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("glearn/io: msgpack load from file failed: %w", err)
	}
	defer f.Close()

	return LoadMsgpack(f, model)
}
