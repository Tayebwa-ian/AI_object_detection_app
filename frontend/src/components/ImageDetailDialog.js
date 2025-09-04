import React from "react";
import { Dialog, DialogTitle, DialogContent, Box, Typography, Chip, Stack } from "@mui/material";

// basic overlay boxes rendering; predictions expected as [{label, confidence, bbox: [x,y,w,h] }]
export default function ImageDetailDialog({ open, onClose, image }) {
  if (!image) return null;

  const { url, filename, predictions = [] } = image;

  // We render the image in a container and overlay boxes scaled to image display size.
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>{filename || image.id}</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', gap: 2, flexDirection: { xs: 'column', md: 'row' } }}>
          <Box sx={{ position: 'relative', width: '100%', maxWidth: 640 }}>
            <Box component="img" src={url} alt={filename} sx={{ width: '100%', height: 'auto', display: 'block' }} />
            {/* overlay: assumes bbox coordinates are relative [x,y,w,h] with values 0..1 */}
            <Box sx={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
              {predictions.map((p, i) => {
                const [x, y, w, h] = p.bbox || [0, 0, 0, 0];
                return (
                  <Box
                    key={i}
                    sx={{
                      position: 'absolute',
                      left: `${x * 100}%`,
                      top: `${y * 100}%`,
                      width: `${w * 100}%`,
                      height: `${h * 100}%`,
                      border: '2px solid rgba(255,0,0,0.8)',
                      boxSizing: 'border-box',
                    }}
                  />
                );
              })}
            </Box>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle1" gutterBottom>
              Predictions
            </Typography>
            <Stack spacing={1}>
              {predictions.length === 0 && <Typography variant="body2">No predictions available.</Typography>}
              {predictions.map((p, i) => (
                <Box key={i} sx={{ border: '1px solid', borderColor: 'divider', p: 1, borderRadius: 1 }}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Typography variant="body2">{p.label}</Typography>
                    <Chip label={`${Math.round(p.confidence * 100)}%`} size="small" />
                  </Stack>
                  <Typography variant="caption" color="text.secondary">
                    bbox: {JSON.stringify(p.bbox)}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
