// src/pages/ModelsPage.js

import React, { useCallback, useEffect, useState } from "react";
import { Box, Typography, Button, TextField } from "@mui/material";
import DataTable from "../components/DataTable";
import Feedback from "../components/Feedback";
import { getModels, createModel, updateModel, deleteModel } from "../api/models";
import { useLoading } from "../context/LoadingContext";

const ModelsPage = () => {
  const { startLoading, stopLoading } = useLoading();
  const [models, setModels] = useState([]);
  const [newModel, setNewModel] = useState({ name: "", description: "" });
  const [feedback, setFeedback] = useState({ open: false, message: "", severity: "success" });

  const fetchModels = useCallback(async () => {
    try {
      startLoading();
      const data = await getModels();
      setModels(data);
    } catch (err) {
      console.error(err);
      setFeedback({ open: true, message: "Failed to load models", severity: "error" });
    } finally {
      stopLoading();
    }
  }, [startLoading, stopLoading]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleAdd = async () => {
    if (!newModel.name) {
      setFeedback({ open: true, message: "Model name is required", severity: "warning" });
      return;
    }
    try {
      startLoading();
      await createModel(newModel);
      setNewModel({ name: "", description: "" });
      await fetchModels();
      setFeedback({ open: true, message: "Model added", severity: "success" });
    } catch (err) {
      console.error(err);
      setFeedback({ open: true, message: "Error adding model", severity: "error" });
    } finally {
      stopLoading();
    }
  };

  const handleUpdate = async (row) => {
    try {
      startLoading();
      await updateModel(row.id, row);
      await fetchModels();
      setFeedback({ open: true, message: "Model updated", severity: "success" });
    } catch (err) {
      console.error(err);
      setFeedback({ open: true, message: "Error updating model", severity: "error" });
    } finally {
      stopLoading();
    }
  };

  const handleDelete = async (id) => {
    try {
      startLoading();
      await deleteModel(id);
      await fetchModels();
      setFeedback({ open: true, message: "Model deleted", severity: "success" });
    } catch (err) {
      console.error(err);
      setFeedback({ open: true, message: "Error deleting model", severity: "error" });
    } finally {
      stopLoading();
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Manage Models
      </Typography>
      <Typography variant="subtitle1" gutterBottom>
        Admin only
      </Typography>

      <Box sx={{ display: "flex", gap: 2, mb: 2, mt: 2 }}>
        <TextField
          label="Name"
          value={newModel.name}
          onChange={(e) => setNewModel({ ...newModel, name: e.target.value })}
        />
        <TextField
          label="Description"
          value={newModel.description}
          onChange={(e) => setNewModel({ ...newModel, description: e.target.value })}
        />
        <Button variant="contained" onClick={handleAdd}>
          Add
        </Button>
      </Box>

      <DataTable
        rows={models}
        columns={[
          { field: "id", headerName: "ID", width: 150 },
          { field: "name", headerName: "Name", width: 200, editable: true },
          { field: "description", headerName: "Description", width: 300, editable: true },
          {
            field: "actions",
            headerName: "Actions",
            width: 140,
            renderCell: (params) => (
              <Box sx={{ display: "flex", gap: 1 }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => handleUpdate(params.row)}
                >
                  Save
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  onClick={() => handleDelete(params.row.id)}
                >
                  Delete
                </Button>
              </Box>
            ),
          },
        ]}
        onUpdate={handleUpdate}
      />

      <Feedback
        open={feedback.open}
        message={feedback.message}
        severity={feedback.severity}
        onClose={() => setFeedback({ ...feedback, open: false })}
      />
    </Box>
  );
};

export default ModelsPage;
