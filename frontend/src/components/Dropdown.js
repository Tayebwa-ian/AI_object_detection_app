// src/components/Dropdown.js
// Generic dropdown for labels, models, etc.

import React from "react";
import { FormControl, InputLabel, Select, MenuItem } from "@mui/material";

const Dropdown = ({ label, value, options, onChange, multiple = false }) => {
  return (
    <FormControl fullWidth sx={{ mt: 2 }}>
      <InputLabel>{label}</InputLabel>
      <Select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        multiple={multiple}
        label={label}
      >
        {options.map((opt) => (
          <MenuItem key={opt.id} value={opt.id}>
            {opt.name}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};

export default Dropdown;
