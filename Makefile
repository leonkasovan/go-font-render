# ------------------------------
# Build configuration
# ------------------------------

GO       := go
BUILD_DATE := $(shell date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown_date)
BIN_DIR  := .
SRC_DIR  := src

# ------------------------------
# Detect platform
# ------------------------------

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
    DEFAULT_TARGET := linux
	ifeq ($(UNAME_M),aarch64)
# 		TAGS := sdl2,opengles31,gles2
		TAGS := x11,opengles31,gles2
# 		TAGS := x11,vulkan
# 		TAGS := wayland,opengles31,gles2
# 		TAGS := opengl21
	else
		TAGS := opengl33
	endif
else ifeq ($(OS),Windows_NT)
    DEFAULT_TARGET := win
	TAGS := opengl33
else
    DEFAULT_TARGET := linux
	TAGS := opengl33
endif

# ------------------------------
# Main targets
# ------------------------------

.PHONY: all win linux clean assets screenpack

all: $(DEFAULT_TARGET)

win:
	@echo "Building for Windows with $(TAGS)..."
	CGO_ENABLED=1 GOEXPERIMENT=arenas GOOS=windows GOARCH=amd64 \
	$(GO) build -tags=$(TAGS) -trimpath -v -ldflags "-s -w" \
	-o $(BIN_DIR)/ikemen_win.exe ./$(SRC_DIR)

linux:
	@echo "Building for Linux with $(TAGS)..."
	CGO_ENABLED=1 GOEXPERIMENT=arenas GOOS=linux \
	$(GO) build -x -tags=$(TAGS) -trimpath -v -ldflags "-s -w" \
	-o $(BIN_DIR)/ikemen_linux ./$(SRC_DIR)
