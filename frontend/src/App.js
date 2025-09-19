// src/App.js

import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { CssBaseline, Container } from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";

import theme from "./layout/Theme";
import AppLayout from "./layout/AppLayout";
import GlobalLoading from "./components/GlobalLoading";
import { LoadingProvider } from "./context/LoadingContext";

import LabelsPage from "./pages/LabelsPage";
import ModelsPage from "./pages/ModelsPage";
import NormalMode from "./pages/NormalMode";
import TrainTestMode from "./pages/TrainTestMode";
import AdvancedMode from "./pages/AdvancedMode";

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <LoadingProvider>
        <Router>
          <AppLayout>
            <Container sx={{ mt: 3 }}>
              <Routes>
                <Route path="/labels" element={<LabelsPage />} />
                <Route path="/models" element={<ModelsPage />} />
                <Route path="/normal" element={<NormalMode />} />
                <Route path="/train" element={<TrainTestMode />} />
                <Route path="/advanced" element={<AdvancedMode />} />
                <Route path="/" element={<NormalMode />} />
              </Routes>
            </Container>
          </AppLayout>
        </Router>
        <GlobalLoading />
      </LoadingProvider>
    </ThemeProvider>
  );
}

export default App;
