# AI Object Counter - Monitoring Stack

This directory contains the monitoring infrastructure for the AI Object Counting Application using Prometheus and Grafana.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backend API   │───▶│   Prometheus    │───▶│     Grafana     │
│  (Flask App)    │    │  (Metrics DB)   │    │  (Dashboards)   │
│  Port: 5000     │    │  Port: 9090     │    │  Port: 3001     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Metrics Collected

### Application Metrics
- **Response Time**: `app_response_seconds` - API request processing time
- **Request Rate**: `api_requests_total` - Number of API requests per endpoint
- **Request Status**: HTTP status codes and error rates

### Model Performance Metrics
- **Inference Time**: `model_inference_seconds` - Time spent by each model (SAM, ResNet, Mapper)
- **Confidence Scores**: `model_confidence` - Model confidence per object type
- **Quality Metrics**: `model_accuracy`, `model_precision`, `model_recall`, `model_f1_score`

### Image Processing Metrics
- **Image Dimensions**: `image_width_pixels`, `image_height_pixels`
- **Object Detection**: `predicted_object_count` - Objects detected per type
- **Segmentation**: `segments_found_total` - Number of segments found
- **Object Types**: `object_types_found_total` - Different object types detected
- **Segment Area**: `avg_segment_area_pixels` - Average segment size

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Backend application running on `localhost:5000`

### Start Monitoring Stack

**Windows:**
```bash
cd monitoring
start-monitoring.bat
```

**Linux/Mac:**
```bash
cd monitoring
chmod +x start-monitoring.sh
./start-monitoring.sh
```

**Manual:**
```bash
cd monitoring
docker-compose up -d
```

### Access Services

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001
  - Username: `admin`
  - Password: `admin123`

## 📈 Grafana Dashboard

The dashboard includes the following panels:

1. **API Response Time** - 95th and 50th percentile response times
2. **Model Inference Time** - Performance of SAM, ResNet, and Mapper models
3. **Model Confidence Scores** - Confidence levels for predictions
4. **Quality Metrics** - Accuracy, Precision, Recall, F1 scores per object type
5. **API Request Rate** - Requests per second by endpoint
6. **Object Detection Metrics** - Predicted counts, segments, object types

## 🔧 Configuration

### Prometheus Configuration (`prometheus.yml`)
- Scrapes backend metrics every 10 seconds
- Stores data for 200 hours
- Targets backend at `host.docker.internal:5000`

### Grafana Configuration
- **Datasource**: Automatically configured to use Prometheus
- **Dashboards**: Pre-provisioned AI Object Counter dashboard
- **Authentication**: Default admin/admin123 (change in production)

## 📋 Useful Commands

```bash
# View logs
docker-compose logs -f

# Stop monitoring stack
docker-compose down

# Restart services
docker-compose restart

# View specific service logs
docker-compose logs -f prometheus
docker-compose logs -f grafana

# Check service status
docker-compose ps
```

## 🔍 Troubleshooting

### Backend Not Reachable
- Ensure backend is running on `localhost:5000`
- Check if `/metrics` endpoint is accessible: `curl http://localhost:5000/metrics`
- Verify firewall settings

### Prometheus Not Scraping
- Check Prometheus targets: http://localhost:9090/targets
- Verify backend is accessible from Docker container
- Check Prometheus logs: `docker-compose logs prometheus`

### Grafana Dashboard Empty
- Verify Prometheus datasource is working
- Check if metrics are being collected: http://localhost:9090/graph
- Ensure backend is processing requests to generate metrics

### Port Conflicts
- Prometheus uses port 9090
- Grafana uses port 3001 (to avoid conflict with frontend on 3000)
- Backend should use port 5000

## 🛠️ Customization

### Adding New Metrics
1. Add metrics to `src/monitoring/metrics.py`
2. Update dashboard JSON in `grafana/dashboards/`
3. Restart monitoring stack

### Modifying Dashboard
1. Access Grafana at http://localhost:3001
2. Edit dashboard: Dashboards > AI Object Counter > Edit
3. Save changes (they will persist in the container)

### Changing Scrape Interval
Edit `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'ai-object-counter-backend'
    scrape_interval: 5s  # Change this value
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenMetrics Format](https://openmetrics.io/)
- [Prometheus Python Client](https://prometheus.github.io/client_python/)
