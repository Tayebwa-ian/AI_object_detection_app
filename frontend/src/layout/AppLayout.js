// src/layout/AppLayout.js
// Provides AppBar + Drawer + consistent layout across pages

import React from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemText,
  CssBaseline,
  Box,
} from "@mui/material";
import { Link } from "react-router-dom";

const drawerWidth = 220;

const AppLayout = ({ children }) => {
  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />

      {/* Top App Bar */}
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Object Detection UI
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Side Drawer Navigation */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          "& .MuiDrawer-paper": { width: drawerWidth, boxSizing: "border-box" },
        }}
      >
        <Toolbar />
        <List>
          <ListItem button component={Link} to="/labels">
            <ListItemText primary="Labels" />
          </ListItem>
          <ListItem button component={Link} to="/models">
            <ListItemText primary="Models" />
          </ListItem>
          <ListItem button component={Link} to="/normal">
            <ListItemText primary="Normal Mode" />
          </ListItem>
          <ListItem button component={Link} to="/train">
            <ListItemText primary="Train/Test Mode" />
          </ListItem>
          <ListItem button component={Link} to="/advanced">
            <ListItemText primary="Advanced Mode" />
          </ListItem>
        </List>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{ flexGrow: 1, bgcolor: "background.default", p: 3, ml: `${drawerWidth}px` }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default AppLayout;
