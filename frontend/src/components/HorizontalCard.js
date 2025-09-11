import React from "react";
import { Card, CardContent, Typography, Divider, Grid, Box } from "@mui/material";
import { Link } from "react-router-dom";

export default function HorizontalCard({ data }) {
  return (
    <Card
      component={Link}
      to={`/history/${data.id}`}
      sx={{
        display: "block",
        textDecoration: "none",
        color: "inherit",
        borderRadius: 2,
        border: "1px solid",
        borderColor: "divider",
        p: 1,
        mb: 2,
        transition: "all 0.16s",
        '&:hover': { boxShadow: 4, transform: 'translateY(-3px)' },
      }}
    >
      <Grid container>
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', flexDirection: 'column' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                {data.title}
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                {data.summary || data.body}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Reported: {data.created_at} • Author: {data.author}
              </Typography>
            </CardContent>
          </Box>
        </Grid>
      </Grid>
    </Card>
  );
}