// central theme for MUI
import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    primary: { main: "#1976d2" },
    secondary: { main: "#9c27b0" },
  },
  components: {
    MuiCard: {
      defaultProps: {
        elevation: 0,
      },
    },
  },
});

export default theme;
