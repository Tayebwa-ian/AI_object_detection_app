import React, { useState, useCallback, useEffect } from "react";
import { Box, Typography, Button, TextField } from "@mui/material";
import DataTable from "../components/DataTable";
import Feedback from "../components/Feedback";
import { getLabels, createLabel, updateLabel, deleteLabel } from "../api/labels";
import { useLoading } from "../context/LoadingContext";

const LabelsPage = () => {
  const { startLoading, stopLoading } = useLoading();
  const [labels, setLabels] = useState([]);
  const [newLabel, setNewLabel] = useState({ name: "", description: "" });
  const [feedback, setFeedback] = useState({ open: false, message: "", severity: "success" });

  // Wrap in useCallback to stabilize function reference
  const fetchLabels = useCallback(async () => {
    try {
      startLoading();
      const data = await getLabels();
      setLabels(data);
    } catch {
      setFeedback({ open: true, message: "Failed to load labels", severity: "error" });
    } finally {
      stopLoading();
    }
  }, [startLoading, stopLoading]);

  useEffect(() => {
    fetchLabels();
  }, [fetchLabels]);

  const handleAdd = async () => {
    try {
      startLoading();
      await createLabel(newLabel);
      setNewLabel({ name: "", description: "" });
      fetchLabels();
      setFeedback({ open: true, message: "Label added", severity: "success" });
    } catch {
      setFeedback({ open: true, message: "Error adding label", severity: "error" });
    } finally {
      stopLoading();
    }
  };

  const handleUpdate = async (row) => {
    try {
      startLoading();
      await updateLabel(row.id, row);
      fetchLabels();
      setFeedback({ open: true, message: "Label updated", severity: "success" });
    } catch {
      setFeedback({ open: true, message: "Error updating label", severity: "error" });
    } finally {
      stopLoading();
    }
  };

  const handleDelete = async (id) => {
    try {
      startLoading();
      await deleteLabel(id);
      fetchLabels();
      setFeedback({ open: true, message: "Label deleted", severity: "success" });
    } catch {
      setFeedback({ open: true, message: "Error deleting label", severity: "error" });
    } finally {
      stopLoading();
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Add New Labels
      </Typography>

      <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
        <TextField
          label="Name"
          value={newLabel.name}
          onChange={(e) => setNewLabel({ ...newLabel, name: e.target.value })}
        />
        <TextField
          label="Description"
          value={newLabel.description}
          onChange={(e) => setNewLabel({ ...newLabel, description: e.target.value })}
        />
        <Button variant="contained" onClick={handleAdd}>
          Add
        </Button>
      </Box>

      <DataTable
        rows={labels}
        columns={[
          { field: "id", headerName: "ID", width: 150 },
          { field: "name", headerName: "Name", width: 150, editable: true },
          { field: "description", headerName: "Description", width: 200, editable: true },
        ]}
        onUpdate={handleUpdate}
        onDelete={handleDelete}
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

export default LabelsPage;
