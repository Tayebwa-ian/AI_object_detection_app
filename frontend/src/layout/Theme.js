// src/layout/theme.js
// Custom MUI theme for consistent look and feel

import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2", // Blue
    },
    secondary: {
      main: "#f50057", // Pink
    },
    background: {
      default: "#f4f6f8",
      paper: "#ffffff",
    },
  },
  typography: {
    fontFamily: "Roboto, sans-serif",
    h6: {
      fontWeight: 600,
    },
    body1: {
      fontSize: 14,
    },
  },
  shape: {
    borderRadius: 12,
  },
});

export default theme;
