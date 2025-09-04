import React, { useState } from "react";
import { Box, TextField, Button, Stack } from "@mui/material";

export default function ObjectForm({ onCreate }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    onCreate({ name: name.trim(), description: description.trim() });
    setName("");
    setDescription("");
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ mb: 2 }}>
      <Stack spacing={2} direction={{ xs: 'column', sm: 'row' }}>
        <TextField label="Object name" value={name} onChange={(e) => setName(e.target.value)} required />
        <TextField label="Description" value={description} onChange={(e) => setDescription(e.target.value)} />
        <Button type="submit" variant="contained">Add</Button>
      </Stack>
    </Box>
  );
}
