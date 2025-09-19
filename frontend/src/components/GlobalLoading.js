// src/components/GlobalLoading.js
// Full-screen loading indicator with backdrop

import React from "react";
import { Backdrop, CircularProgress } from "@mui/material";
import { useLoading } from "../context/LoadingContext";

const GlobalLoading = () => {
  const { loading } = useLoading();

  return (
    <Backdrop
      sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }}
      open={loading}
    >
      <CircularProgress color="inherit" />
    </Backdrop>
  );
};

export default GlobalLoading;
