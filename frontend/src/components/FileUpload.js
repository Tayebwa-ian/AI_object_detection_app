import React, { useRef, useState } from "react";
import { Box, Button, IconButton, LinearProgress, Typography } from "@mui/material";
import { styled } from "@mui/material/styles";
import DeleteIcon from "@mui/icons-material/Delete";

// Thumbnail styling
const Thumbnail = styled("img")({
  width: "256px",
  height: "256px",
  objectFit: "cover",
  borderRadius: "4px",
  margin: "5px",
});

const FileUpload = ({ files, setFiles }) => {
  const inputRef = useRef();
  const [progresses, setProgresses] = useState({});

  const handleFiles = (selectedFiles) => {
    const newFiles = [...files, ...selectedFiles];
    setFiles(newFiles);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith("image/"));
    handleFiles(droppedFiles);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleRemove = (index) => {
    const newFiles = files.filter((_, i) => i !== index);
    setFiles(newFiles);
    setProgresses((prev) => {
      const copy = { ...prev };
      delete copy[index];
      return copy;
    });
  };

  // Simulate upload with progress
  const simulateUpload = (file, index) => {
    return new Promise((resolve) => {
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.floor(Math.random() * 10) + 5;
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          resolve();
        }
        setProgresses((prev) => ({ ...prev, [index]: progress }));
      }, 200);
    });
  };

  const handleUploadAll = async () => {
    for (let i = 0; i < files.length; i++) {
      await simulateUpload(files[i], i);
    }
  };

  return (
    <Box sx={{ mt: 2 }}>
      <Box
        sx={{
          border: "2px dashed #ccc",
          padding: 2,
          borderRadius: 2,
          textAlign: "center",
          cursor: "pointer",
          mb: 2,
        }}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => inputRef.current.click()}
      >
        <Typography>Drag & Drop images here or click to select</Typography>
      </Box>

      <input
        type="file"
        accept="image/*"
        multiple
        style={{ display: "none" }}
        ref={inputRef}
        onChange={(e) => handleFiles(Array.from(e.target.files))}
      />

      {files.length > 0 && (
        <Box sx={{ display: "flex", flexWrap: "wrap" }}>
          {files.map((file, idx) => (
            <Box key={idx} sx={{ position: "relative" }}>
              <Thumbnail src={URL.createObjectURL(file)} alt={file.name} />
              <IconButton
                onClick={() => handleRemove(idx)}
                sx={{
                  position: "absolute",
                  top: 0,
                  right: 0,
                  backgroundColor: "rgba(255,255,255,0.7)",
                }}
              >
                <DeleteIcon />
              </IconButton>
              {progresses[idx] !== undefined && (
                <Box sx={{ width: "256px", mt: 1 }}>
                  <LinearProgress variant="determinate" value={progresses[idx]} />
                  <Typography variant="caption">{progresses[idx]}%</Typography>
                </Box>
              )}
            </Box>
          ))}
        </Box>
      )}

      {files.length > 0 && (
        <Button variant="contained" sx={{ mt: 2 }} onClick={handleUploadAll}>
          Upload All
        </Button>
      )}
    </Box>
  );
};

export default FileUpload;
