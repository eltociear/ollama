package llm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

type GGML struct {
	ModelType string

	magic uint32
	container
	hyperparameters any

	// fields elevated from hyperparameters
	NumVocab uint32
	FileType ggmlFileType
}

type ggmlFileType uint32

func (ft ggmlFileType) String() string {
	switch ft {
	case ggmlFileType(GGML_FTYPE_ALL_F32):
		return "F32"
	case ggmlFileType(GGML_FTYPE_MOSTLY_F16):
		return "F16"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q4_0):
		return "Q4_0"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q4_1):
		return "Q4_1"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q4_1_SOME_F16):
		return "Q4_1_SOME_F16"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q8_0):
		return "Q8_0"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q5_0):
		return "Q5_0"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q5_1):
		return "Q5_1"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q2_K):
		return "Q2_K"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q3_K):
		return "Q3_K"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q4_K):
		return "Q4_K"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q5_K):
		return "Q5_K"
	case ggmlFileType(GGML_FTYPE_MOSTLY_Q6_K):
		return "Q6_K"
	default:
		return "U"
	}
}

type container interface {
	Name() string
	Decode(io.Reader) error
}

type containerGGML struct {
}

func (c *containerGGML) Name() string {
	return "ggml"
}

func (c *containerGGML) Decode(r io.Reader) error {
	return nil
}

type containerGGMF struct {
	version uint32
}

func (c *containerGGMF) Name() string {
	return "ggmf"
}

func (c *containerGGMF) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

type containerGGJT struct {
	version uint32
}

func (c *containerGGJT) Name() string {
	return "ggjt"
}

func (c *containerGGJT) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1, 2, 3:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

type containerLORA struct {
	version uint32
}

func (c *containerLORA) Name() string {
	return "ggla"
}

func (c *containerLORA) Decode(r io.Reader) error {
	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return errors.New("invalid version")
	}

	c.version = version
	return nil
}

const (
	// / Magic constant for `ggml` files (unversioned).
	FILE_MAGIC_GGML = 0x67676d6c
	// / Magic constant for `ggml` files (versioned, ggmf).
	FILE_MAGIC_GGMF = 0x67676d66
	// / Magic constant for `ggml` files (versioned, ggjt).
	FILE_MAGIC_GGJT = 0x67676a74
	// / Magic constant for `ggla` files (LoRA adapter).
	FILE_MAGIC_GGLA = 0x67676C61
)

const (
	GGML_FTYPE_ALL_F32 = iota
	GGML_FTYPE_MOSTLY_F16
	GGML_FTYPE_MOSTLY_Q4_0
	GGML_FTYPE_MOSTLY_Q4_1
	GGML_FTYPE_MOSTLY_Q4_1_SOME_F16
	GGML_FTYPE_MOSTLY_Q8_0 = iota + 3
	GGML_FTYPE_MOSTLY_Q5_0
	GGML_FTYPE_MOSTLY_Q5_1
	GGML_FTYPE_MOSTLY_Q2_K
	GGML_FTYPE_MOSTLY_Q3_K
	GGML_FTYPE_MOSTLY_Q4_K
	GGML_FTYPE_MOSTLY_Q5_K
	GGML_FTYPE_MOSTLY_Q6_K
	GGML_FTYPE_UNKNOWN = -1
)

func DecodeGGML(r io.ReadSeeker, hint string) (*GGML, error) {
	var ggml GGML
	binary.Read(r, binary.LittleEndian, &ggml.magic)

	switch ggml.magic {
	case FILE_MAGIC_GGML:
		ggml.container = &containerGGML{}
	case FILE_MAGIC_GGMF:
		ggml.container = &containerGGMF{}
	case FILE_MAGIC_GGJT:
		ggml.container = &containerGGJT{}
	case FILE_MAGIC_GGLA:
		ggml.container = &containerLORA{}
	default:
		return nil, errors.New("invalid file magic")
	}

	if err := ggml.container.Decode(r); err != nil {
		return nil, err
	}

	// different model types may have different layouts for hyperparameters
	switch hint {
	case "llama":
		var hp llamaHyperparameters
		binary.Read(r, binary.LittleEndian, &hp)

		ggml.hyperparameters = &hp
		ggml.NumVocab = hp.NumVocab
		ggml.FileType = ggmlFileType(hp.FileType)
	default:
		return nil, fmt.Errorf("unsupported model type: %s", hint)
	}

	// final model type
	ggml.ModelType = hint
	return &ggml, nil
}
