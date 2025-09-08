import React from 'react';
import { Alert, AlertTitle, Button, Collapse, Typography, Box } from '@mui/material';
import { ExpandMore, ExpandLess, Refresh, BugReport } from '@mui/icons-material';

interface ErrorDisplayProps {
  error: string | null;
  title?: string;
  severity?: 'error' | 'warning' | 'info';
  onRetry?: () => void;
  onDismiss?: () => void;
  showDetails?: boolean;
  details?: string;
  retryable?: boolean;
}

export function ErrorDisplay({
  error,
  title = 'Error',
  severity = 'error',
  onRetry,
  onDismiss,
  showDetails = false,
  details,
  retryable = false
}: ErrorDisplayProps) {
  const [expanded, setExpanded] = React.useState(false);

  if (!error) return null;

  const getErrorIcon = () => {
    switch (severity) {
      case 'warning':
        return '⚠️';
      case 'info':
        return 'ℹ️';
      default:
        return '❌';
    }
  };

  const getErrorColor = () => {
    switch (severity) {
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'error';
    }
  };

  const formatErrorMessage = (errorMsg: string) => {
    // Common error patterns and their user-friendly messages
    const errorMappings: { [key: string]: string } = {
      'No image file provided': 'Please select an image file to upload',
      'object_type is required': 'Please select the type of object to count',
      'Invalid file type': 'Please upload a valid image file (PNG, JPG, JPEG, BMP, GIF)',
      'File too large': 'Please upload an image smaller than 10MB',
      'AI models are not available': 'The AI processing system is currently unavailable. Please try again later.',
      'Insufficient memory for processing': 'The image is too large or complex for processing. Try a smaller image.',
      'Processing timeout': 'The image took too long to process. Try a smaller or simpler image.',
      'Health check failed': 'Unable to connect to the server. Please check your connection.',
      'Failed to get object types': 'Unable to load object types. Please refresh the page.',
      'Record not found': 'The requested data could not be found.',
      'Database operation failed': 'There was an error accessing the database. Please try again.',
      'Validation failed': 'Please check your input and try again.',
      'Network error': 'Please check your internet connection and try again.',
      'Server error': 'The server encountered an error. Please try again later.'
    };

    // Check for exact matches first
    if (errorMappings[errorMsg]) {
      return errorMappings[errorMsg];
    }

    // Check for partial matches
    for (const [key, value] of Object.entries(errorMappings)) {
      if (errorMsg.toLowerCase().includes(key.toLowerCase())) {
        return value;
      }
    }

    // Return original message if no mapping found
    return errorMsg;
  };

  const getErrorCategory = (errorMsg: string) => {
    if (errorMsg.toLowerCase().includes('network') || errorMsg.toLowerCase().includes('connection')) {
      return 'Network';
    } else if (errorMsg.toLowerCase().includes('validation') || errorMsg.toLowerCase().includes('required')) {
      return 'Input';
    } else if (errorMsg.toLowerCase().includes('processing') || errorMsg.toLowerCase().includes('ai')) {
      return 'Processing';
    } else if (errorMsg.toLowerCase().includes('database') || errorMsg.toLowerCase().includes('storage')) {
      return 'Database';
    } else if (errorMsg.toLowerCase().includes('server') || errorMsg.toLowerCase().includes('internal')) {
      return 'Server';
    }
    return 'General';
  };

  const errorCategory = getErrorCategory(error);
  const friendlyMessage = formatErrorMessage(error);

  return (
    <Alert 
      severity={getErrorColor() as any}
      sx={{ 
        mb: 2,
        '& .MuiAlert-message': {
          width: '100%'
        }
      }}
      action={
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {retryable && onRetry && (
            <Button
              size="small"
              startIcon={<Refresh />}
              onClick={onRetry}
              variant="outlined"
              color="inherit"
            >
              Retry
            </Button>
          )}
          {onDismiss && (
            <Button
              size="small"
              onClick={onDismiss}
              variant="text"
              color="inherit"
            >
              Dismiss
            </Button>
          )}
        </Box>
      }
    >
      <AlertTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {getErrorIcon()} {title}
      </AlertTitle>
      
      <Typography variant="body2" sx={{ mb: 1 }}>
        {friendlyMessage}
      </Typography>

      {errorCategory !== 'General' && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Category: {errorCategory}
        </Typography>
      )}

      {(showDetails || details) && (
        <>
          <Button
            size="small"
            startIcon={expanded ? <ExpandLess /> : <ExpandMore />}
            onClick={() => setExpanded(!expanded)}
            sx={{ mb: 1 }}
          >
            {expanded ? 'Hide' : 'Show'} Details
          </Button>
          
          <Collapse in={expanded}>
            <Box sx={{ 
              p: 2, 
              bgcolor: 'background.paper', 
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'divider'
            }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                <BugReport sx={{ fontSize: 16, mr: 0.5, verticalAlign: 'middle' }} />
                Technical Details:
              </Typography>
              <Typography 
                variant="body2" 
                component="pre" 
                sx={{ 
                  fontFamily: 'monospace',
                  fontSize: '0.75rem',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  color: 'text.secondary'
                }}
              >
                {details || error}
              </Typography>
            </Box>
          </Collapse>
        </>
      )}

      {errorCategory === 'Network' && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          💡 Try refreshing the page or checking your internet connection
        </Typography>
      )}

      {errorCategory === 'Input' && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          💡 Please check your input and try again
        </Typography>
      )}

      {errorCategory === 'Processing' && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          💡 Try using a smaller or simpler image
        </Typography>
      )}
    </Alert>
  );
}

export default ErrorDisplay;








