import React from "react";
import { Grid, Card, CardMedia, CardActionArea, CardContent, Typography } from "@mui/material";

export default function ImageGrid({ images = [], onImageClick }) {
  return (
    <Grid container spacing={2}>
      {images.map((img) => (
        <Grid item key={img.id} xs={12} sm={6} md={4} lg={3}>
          <Card>
            <CardActionArea onClick={() => onImageClick(img)}>
              <CardMedia component="img" height="160" image={img.url} alt={img.filename || img.id} />
              <CardContent>
                <Typography variant="body2" noWrap>
                  {img.filename || img.id}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {img.prediction ? `${img.prediction.length || 0} objects` : "No predictions"}
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}
