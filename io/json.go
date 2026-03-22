package io

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
)

// SaveJSON saves a model to a writer in JSON format with CRC32 checksum.
// The model is first serialized to JSON bytes (inner), then wrapped in an
// Envelope with integrity metadata and serialized again (outer).
func SaveJSON(w io.Writer, model any) error {
	// Serialize the model to JSON bytes (inner payload).
	data, err := json.Marshal(model)
	if err != nil {
		return fmt.Errorf("glearn/io: JSON save failed: could not marshal model: %w", err)
	}

	// Build envelope with checksum.
	env := Envelope{
		Format:   "json",
		Type:     reflect.TypeOf(model).String(),
		Version:  Version,
		Checksum: computeChecksum(data),
		Data:     data,
	}

	// Serialize the envelope to the writer.
	encoder := json.NewEncoder(w)
	if err := encoder.Encode(env); err != nil {
		return fmt.Errorf("glearn/io: JSON save failed: could not encode envelope: %w", err)
	}

	return nil
}

// LoadJSON loads a model from a reader, verifying the CRC32 checksum.
// The model parameter must be a pointer to the target type.
func LoadJSON(r io.Reader, model any) error {
	// Decode the envelope.
	var env Envelope
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&env); err != nil {
		return fmt.Errorf("glearn/io: JSON load failed: could not decode envelope: %w", err)
	}

	// Verify checksum.
	if !verifyChecksum(env.Data, env.Checksum) {
		return fmt.Errorf("glearn/io: JSON load failed: checksum mismatch")
	}

	// Deserialize the model from the inner data.
	if err := json.Unmarshal(env.Data, model); err != nil {
		return fmt.Errorf("glearn/io: JSON load failed: could not unmarshal model: %w", err)
	}

	return nil
}

// SaveJSONFile saves a model to a file in JSON format.
func SaveJSONFile(path string, model any) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("glearn/io: JSON save to file failed: %w", err)
	}
	defer f.Close()

	if err := SaveJSON(f, model); err != nil {
		return err
	}

	return f.Close()
}

// LoadJSONFile loads a model from a JSON file.
func LoadJSONFile(path string, model any) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("glearn/io: JSON load from file failed: %w", err)
	}
	defer f.Close()

	return LoadJSON(f, model)
}
