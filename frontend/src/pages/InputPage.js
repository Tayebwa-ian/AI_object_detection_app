import React, { useState } from 'react';
import { Box, Container, Typography, Button, TextField, Stack } from '@mui/material';
import ImageGrid from '../components/ImageGrid';
import ImageDetailDialog from '../components/ImageDetailDialog';
import DataLoading from '../components/DataLoading';
import ErrorDisplay from '../components/ErrorDisplay';
import { sendJSON } from '../CRUD/postUpdateDelete';

export default function InputPage() {
  const [files, setFiles] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [results, setResults] = useState([]); // array of image objects with predictions
  const [selected, setSelected] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFiles = (e) => {
    const f = Array.from(e.target.files).map((file, i) => ({ id: `${Date.now()}-${i}`, file, filename: file.name, url: URL.createObjectURL(file) }));
    setFiles(f);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    if (files.length === 0) { setError('Please select one or more images.'); return; }

    setIsLoading(true);
    try {
      // send files + prompt to backend. Example uses FormData.
      const fd = new FormData();
      files.forEach((f) => fd.append('images', f.file));
      fd.append('prompt', prompt);

      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Prediction failed');
      }
      const data = await res.json();
      // Expecting data.items -> array of {id, url, prediction: [{label, confidence, bbox}] }
      setResults(data.items || []);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) return <DataLoading message="Running model on images..." />;

  return (
    <Container sx={{ py: 4 }}>
      <Typography variant="h5" gutterBottom>Submit images for object detection</Typography>
      <Box component="form" onSubmit={handleSubmit} sx={{ mb: 3 }}>
        <Stack spacing={2}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <Button variant="outlined" component="label">Select Images
              <input hidden accept="image/*" multiple type="file" onChange={handleFiles} />
            </Button>
            <TextField value={prompt} onChange={(e) => setPrompt(e.target.value)} fullWidth placeholder="Enter any extra prompt/context for the model" />
            <Button type="submit" variant="contained">Submit</Button>
          </Stack>
          <ErrorDisplay error={error} />
        </Stack>
      </Box>

      {results.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>Results</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Submitted: {results.length} images — click any to view predictions
          </Typography>

          <ImageGrid images={results} onImageClick={(img) => setSelected(img)} />
        </Box>
      )}

      <ImageDetailDialog open={Boolean(selected)} image={selected} onClose={() => setSelected(null)} />
    </Container>
  );
}
