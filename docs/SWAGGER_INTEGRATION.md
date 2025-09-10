# Swagger API Documentation Integration

## Overview

The AI Object Counter API now includes comprehensive Swagger/OpenAPI documentation for all endpoints. This provides interactive API documentation, request/response examples, and testing capabilities.

## Accessing Swagger Documentation

### Local Development
When running the application locally, access Swagger UI at:
```
http://localhost:5000/apidocs/
```

### Production
In production, access Swagger UI at:
```
https://your-domain.com/apidocs/
```

## Features

### **Complete API Coverage**
All API endpoints are documented with:
- **Request/Response schemas**
- **Parameter descriptions**
- **Example values**
- **Error responses**
- **HTTP status codes**

### **Interactive Testing**
- Test endpoints directly from the browser
- Upload files for testing
- View real-time responses
- No need for external tools like Postman

### **Comprehensive Schemas**
- Detailed data models for all request/response objects
- Validation rules and constraints
- Example data for all fields

## API Endpoints Documentation

### **Inputs** - Image Upload and Processing
- `POST /api/count` - Upload image and count objects
- `POST /api/count-all` - Auto-detect all objects in image
- `GET /api/inputs` - List all input records

### **Outputs** - Prediction Results and Corrections
- `GET /api/results` - List all prediction results
- `GET /api/results/{id}` - Get specific result
- `PUT /api/correct/{id}` - Submit correction for prediction
- `DELETE /api/results/{id}` - Delete result

### **Object Types** - Object Type Management
- `GET /api/object-types` - List all object types
- `POST /api/object-types` - Create new object type
- `GET /api/object/{id}` - Get specific object type
- `PUT /api/object/{id}` - Update object type
- `DELETE /api/object/{id}` - Delete object type

### **Batch Processing** - Batch Operations
- `POST /api/batch/process` - Process multiple images
- `GET /api/batch/status` - Get batch processing statistics

### **Few-Shot Learning** - Custom Object Types
- `POST /api/fewshot/register` - Register new object type with few-shot learning
- `POST /api/fewshot/count` - Count objects using few-shot learning
- `GET /api/fewshot/object-types` - List few-shot object types
- `GET /api/fewshot/object-types/{name}` - Get specific few-shot object type
- `DELETE /api/fewshot/object-types/{name}` - Delete few-shot object type
- `GET /api/fewshot/predictions` - Get recent few-shot predictions

### **Performance Monitoring** - System Metrics
- `GET /api/performance/metrics` - Get performance metrics
- `GET /api/performance/object-types` - Get object type statistics
- `GET /api/performance/database` - Get database statistics
- `POST /api/performance/reset` - Reset performance statistics
- `GET /api/performance/health` - Get comprehensive system health

### **System** - System Endpoints
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /media/{filename}` - Serve media files

## Data Models

### Core Models
- **Input** - Image upload records
- **Output** - Prediction results
- **ObjectType** - Object type definitions

### Request/Response Models
- **CountResponse** - Object counting results
- **CorrectionRequest/Response** - Correction submissions
- **BatchProcessingRequest/Response** - Batch operations
- **FewShotRegisterRequest/Response** - Few-shot learning
- **PerformanceMetrics** - System performance data
- **ErrorResponse** - Error handling
- **HealthResponse** - System health status

## Usage Examples

### 1. Upload and Count Objects
```bash
# Using curl
curl -X POST "http://localhost:5000/api/count" \
  -F "image=@car.jpg" \
  -F "object_type=car" \
  -F "description=Counting cars in parking lot"
```

### 2. Submit Correction
```bash
curl -X PUT "http://localhost:5000/api/correct/result-id" \
  -H "Content-Type: application/json" \
  -d '{"corrected_count": 5}'
```

### 3. Batch Processing
```bash
curl -X POST "http://localhost:5000/api/batch/process" \
  -F "images[]=@image1.jpg" \
  -F "images[]=@image2.jpg" \
  -F "object_type=car"
```

### 4. Few-Shot Learning Registration
```bash
curl -X POST "http://localhost:5000/api/fewshot/register" \
  -F "object_name=elephant" \
  -F "support_images[]=@elephant1.jpg" \
  -F "support_images[]=@elephant2.jpg" \
  -F "support_images[]=@elephant3.jpg"
```

## Testing with Swagger UI

### 1. **Navigate to Swagger UI**
Go to `http://localhost:5000/apidocs/`

### 2. **Expand Endpoint**
Click on any endpoint to see its documentation

### 3. **Try It Out**
Click "Try it out" button to enable testing

### 4. **Fill Parameters**
- For file uploads: Use the file picker
- For JSON data: Enter JSON in the request body
- For path parameters: Enter values in the path

### 5. **Execute**
Click "Execute" to send the request

### 6. **View Response**
See the response data, status code, and headers

## Error Handling

All endpoints include comprehensive error documentation:
- **400** - Bad Request (validation errors)
- **404** - Not Found (resource doesn't exist)
- **500** - Internal Server Error (processing errors)

Error responses follow a consistent format:
```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "status": "fail",
  "details": {}
}
```

## Security

The API includes security definitions for:
- **Bearer Token** authentication (for future use)
- **CORS** support for cross-origin requests

## Monitoring Integration

Swagger documentation integrates with the monitoring system:
- **Performance metrics** are tracked for all documented endpoints
- **Response times** are measured and reported
- **Error rates** are monitored and displayed

## Development Workflow

### Adding New Endpoints
1. **Add Swagger docstring** to your endpoint method
2. **Define request/response schemas** in `swagger_template.py`
3. **Test in Swagger UI** to ensure documentation is accurate
4. **Update this documentation** if needed

### Example Swagger Docstring
```python
def post(self):
    """
    Upload image and process with AI pipeline
    ---
    tags:
      - Inputs
    parameters:
      - in: formData
        name: image
        type: file
        required: true
        description: Image file to process
    responses:
      201:
        description: Image processed successfully
        schema:
          $ref: '#/definitions/CountResponse'
      400:
        description: Bad request
        schema:
          $ref: '#/definitions/ErrorResponse'
    """
```

## Benefits

### For Developers
- **Self-documenting API** - No need to maintain separate documentation
- **Interactive testing** - Test endpoints without external tools
- **Consistent schemas** - Clear data models and validation rules
- **Error handling** - Comprehensive error documentation

### For Users
- **Easy integration** - Clear examples and schemas
- **Real-time testing** - Try endpoints before implementing
- **Comprehensive coverage** - All endpoints documented
- **Professional appearance** - Clean, organized documentation

## Troubleshooting

### Swagger UI Not Loading
1. Check if the application is running
2. Verify the URL: `http://localhost:5000/apidocs/`
3. Check browser console for errors
4. Ensure flasgger is installed: `pip install flasgger`

### Missing Endpoints
1. Check if the endpoint has a Swagger docstring
2. Verify the endpoint is registered in `app.py`
3. Restart the application after adding documentation

### Schema Errors
1. Check the schema definitions in `swagger_template.py`
2. Ensure all referenced schemas exist
3. Validate JSON syntax in schema definitions

## Conclusion

The Swagger integration provides a comprehensive, interactive API documentation system that makes the AI Object Counter API easy to understand, test, and integrate with. All endpoints are fully documented with examples, schemas, and error handling information.
