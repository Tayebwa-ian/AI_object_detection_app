import React from 'react';
import { AppBar, Toolbar, Typography, Button } from '@mui/material';
import { Link } from 'react-router-dom';

export default function Header() {
  return (
    <AppBar position="static" color="default" elevation={1} sx={{ mb: 2 }}>
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>AI Image Labeler</Typography>
        <Button component={Link} to="/" color="inherit">Input</Button>
        <Button component={Link} to="/correct" color="inherit">Correct</Button>
        <Button component={Link} to="/history" color="inherit">History</Button>
        <Button component={Link} to="/objects" color="inherit">Objects</Button>
      </Toolbar>
    </AppBar>
  );
}
