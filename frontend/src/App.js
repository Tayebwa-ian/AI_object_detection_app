import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from './theme';
import Header from './components/Header';
import InputPage from './pages/InputPage';
import CorrectPage from './pages/CorrectPage';
import HistoryPage from './pages/HistoryPage';
import ObjectTypePage from './pages/ObjectTypePage';

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Header />
      <Routes>
        <Route path="/" element={<InputPage />} />
        <Route path="/correct" element={<CorrectPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/objects" element={<ObjectTypePage />} />
        <Route path="/history/:id" element={<HistoryPage />} />
      </Routes>
    </ThemeProvider>
  );
}
