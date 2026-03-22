package io

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"os"
	"reflect"
)

// SaveGob saves a model to a writer using Go's gob encoding with CRC32 checksum.
// The model is first gob-encoded to bytes (inner), then wrapped in an Envelope
// and gob-encoded again (outer).
func SaveGob(w io.Writer, model any) error {
	// Serialize the model to gob bytes (inner payload).
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("glearn/io: gob save failed: could not encode model: %w", err)
	}

	data := buf.Bytes()

	// Build envelope with checksum.
	env := Envelope{
		Format:   "gob",
		Type:     reflect.TypeOf(model).String(),
		Version:  Version,
		Checksum: computeChecksum(data),
		Data:     data,
	}

	// Serialize the envelope to the writer.
	envEncoder := gob.NewEncoder(w)
	if err := envEncoder.Encode(env); err != nil {
		return fmt.Errorf("glearn/io: gob save failed: could not encode envelope: %w", err)
	}

	return nil
}

// LoadGob loads a model from gob format, verifying the CRC32 checksum.
// The model parameter must be a pointer to the target type.
func LoadGob(r io.Reader, model any) error {
	// Decode the envelope.
	var env Envelope
	decoder := gob.NewDecoder(r)
	if err := decoder.Decode(&env); err != nil {
		return fmt.Errorf("glearn/io: gob load failed: could not decode envelope: %w", err)
	}

	// Verify checksum.
	if !verifyChecksum(env.Data, env.Checksum) {
		return fmt.Errorf("glearn/io: gob load failed: checksum mismatch")
	}

	// Deserialize the model from the inner data.
	buf := bytes.NewReader(env.Data)
	modelDecoder := gob.NewDecoder(buf)
	if err := modelDecoder.Decode(model); err != nil {
		return fmt.Errorf("glearn/io: gob load failed: could not decode model: %w", err)
	}

	return nil
}

// SaveGobFile saves a model to a file using gob encoding.
func SaveGobFile(path string, model any) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("glearn/io: gob save to file failed: %w", err)
	}
	defer f.Close()

	if err := SaveGob(f, model); err != nil {
		return err
	}

	return f.Close()
}

// LoadGobFile loads a model from a gob-encoded file.
func LoadGobFile(path string, model any) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("glearn/io: gob load from file failed: %w", err)
	}
	defer f.Close()

	return LoadGob(f, model)
}
