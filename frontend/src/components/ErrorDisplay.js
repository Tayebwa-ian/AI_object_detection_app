import React from "react";
import { WarningAmberOutlined } from "@mui/icons-material";
import { Alert } from "@mui/material";

export default function ErrorDisplay({ error, severity = "error" }) {
  if (!error) return null;
  return (
    <Alert icon={<WarningAmberOutlined fontSize="inherit" />} severity={severity} sx={{ my: 2 }}>
      {error}
    </Alert>
  );
}
