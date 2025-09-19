// src/components/MetricsChart.js
// Reusable chart for displaying training/testing metrics

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card, CardContent, Typography } from "@mui/material";

const MetricsChart = ({ metrics }) => {
  if (!metrics) return null;

  // Transform metrics object into chart data
  const data = Object.keys(metrics).map((key) => ({
    metric: key,
    value: metrics[key],
  }));

  return (
    <Card sx={{ mt: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
            <Legend />
            <Bar dataKey="value" fill="#1976d2" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default MetricsChart;
