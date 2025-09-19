// src/layout/AppLayout.js

import React from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  CssBaseline,
  Box,
} from "@mui/material";
import { Link, useLocation } from "react-router-dom";
import LabelIcon from "@mui/icons-material/Label";
import StorageIcon from "@mui/icons-material/Storage";
import CategoryIcon from "@mui/icons-material/Category";
import SchoolIcon from "@mui/icons-material/School";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";

const drawerWidth = 220;

const navItems = [
  { text: "Labels", path: "/labels", icon: <LabelIcon /> },
  { text: "Models", path: "/models", icon: <StorageIcon /> },
  { text: "Normal Mode", path: "/normal", icon: <CategoryIcon /> },
  { text: "Train/Test Mode", path: "/train", icon: <SchoolIcon /> },
  { text: "Advanced Mode", path: "/advanced", icon: <AutoAwesomeIcon /> },
];

const AppLayout = ({ children }) => {
  const location = useLocation();

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />

      {/* Top App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          bgcolor: "primary.main",
        }}
      >
        <Toolbar>
          <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{ fontWeight: "bold", letterSpacing: 1 }}
          >
            Object Counting App
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Side Drawer Navigation */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: drawerWidth,
            boxSizing: "border-box",
            backgroundColor: "background.paper",
            borderRight: "1px solid #ddd",
          },
        }}
      >
        <Toolbar />
        <List>
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <ListItemButton
                key={item.text}
                component={Link}
                to={item.path}
                sx={{
                  borderRadius: 1,
                  mx: 1,
                  mb: 1,
                  backgroundColor: isActive ? "primary.light" : "transparent",
                  "&:hover": {
                    backgroundColor: isActive
                      ? "primary.light"
                      : "action.hover",
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    color: isActive ? "primary.main" : "text.secondary",
                    minWidth: 40,
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.text}
                  primaryTypographyProps={{
                    fontWeight: isActive ? "bold" : "normal",
                    color: isActive ? "primary.main" : "text.primary",
                  }}
                />
              </ListItemButton>
            );
          })}
        </List>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          bgcolor: "background.default",
          p: 3,
          ml: `${drawerWidth}px`,
          minHeight: "100vh",
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default AppLayout;
