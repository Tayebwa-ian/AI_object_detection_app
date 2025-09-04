import React, { useState, useEffect } from 'react';
import { Container, Typography, Button, Box, TextField, Stack } from '@mui/material';
import { useGet } from '../hooks/useGet';
import DataLoading from '../components/DataLoading';
import ErrorDisplay from '../components/ErrorDisplay';
import ImageGrid from '../components/ImageGrid';
import ImageDetailDialog from '../components/ImageDetailDialog';
import { sendJSON } from '../CRUD/postUpdateDelete';

export default function CorrectPage() {
  const { data, isPending, error } = useGet('/api/submissions');
  const [selected, setSelected] = useState(null);
  const [localEdits, setLocalEdits] = useState({});

  if (isPending) return <DataLoading />;
  if (error) return <ErrorDisplay error={error} />;

  const images = (data && data.items) || [];

  const handleSaveCorrection = async (imgId) => {
    // payload contains corrections for image
    const payload = localEdits[imgId] || {};
    try {
      await sendJSON(`/api/submissions/${imgId}/correction`, payload, 'POST');
      alert('Saved');
    } catch (err) { alert(err.message); }
  };

  return (
    <Container sx={{ py: 4 }}>
      <Typography variant="h5" gutterBottom>Review & Correct Predictions</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Click an image to view details and correct counts/labels.
      </Typography>

      <ImageGrid images={images} onImageClick={(img) => setSelected(img)} />

      <ImageDetailDialog open={Boolean(selected)} image={selected} onClose={() => setSelected(null)} />

      {/* Small correction form example */}
      {selected && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="h6">Make corrections for {selected.filename}</Typography>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mt: 1 }}>
            <TextField label="Correct object count" type="number" onChange={(e) => setLocalEdits(prev => ({ ...prev, [selected.id]: { count: Number(e.target.value) } }))} />
            <Button variant="contained" onClick={() => handleSaveCorrection(selected.id)}>Save correction</Button>
          </Stack>
        </Box>
      )}
    </Container>
  );
}
