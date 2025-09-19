// src/routes.js
// Central place for defining routes

import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import LabelsPage from "./pages/LabelsPage";
import ModelsPage from "./pages/ModelsPage";
import NormalMode from "./pages/NormalMode";
import TrainTestMode from "./pages/TrainTestMode";
import AdvancedMode from "./pages/AdvancedMode";

const AppRoutes = () => (
  <Routes>
    <Route path="/" element={<Navigate to="/labels" />} />
    <Route path="/labels" element={<LabelsPage />} />
    <Route path="/models" element={<ModelsPage />} />
    <Route path="/normal" element={<NormalMode />} />
    <Route path="/train" element={<TrainTestMode />} />
    <Route path="/advanced" element={<AdvancedMode />} />
  </Routes>
);

export default AppRoutes;
