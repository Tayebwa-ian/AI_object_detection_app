#!/usr/bin/python3
"""
Performance Monitoring Views module
"""
from flask_restful import Resource
from flask import request, jsonify, make_response
from ...storage import database, Output, Input, ObjectType
import time
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

class PerformanceMonitoring:
    """Performance monitoring singleton"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerformanceMonitoring, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.session_start_time = time.time()
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.processing_times = []
            self.object_type_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'successes': 0, 'failures': 0})
            self._initialized = True
    
    def record_request(self, object_type: str, processing_time: float, success: bool):
        """Record a processing request"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.processing_times.append(processing_time)
        
        # Update object type statistics
        self.object_type_stats[object_type]['count'] += 1
        self.object_type_stats[object_type]['total_time'] += processing_time
        if success:
            self.object_type_stats[object_type]['successes'] += 1
        else:
            self.object_type_stats[object_type]['failures'] += 1
    
    def get_metrics(self):
        """Get current performance metrics"""
        uptime = time.time() - self.session_start_time
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0
        median_processing_time = statistics.median(self.processing_times) if self.processing_times else 0
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate_percent': round(success_rate, 2),
            'average_processing_time': round(avg_processing_time, 3),
            'median_processing_time': round(median_processing_time, 3),
            'min_processing_time': round(min(self.processing_times), 3) if self.processing_times else 0,
            'max_processing_time': round(max(self.processing_times), 3) if self.processing_times else 0,
            'requests_per_minute': round(self.total_requests / (uptime / 60), 2) if uptime > 0 else 0
        }
    
    def get_object_type_stats(self):
        """Get statistics by object type"""
        stats = {}
        for obj_type, data in self.object_type_stats.items():
            avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
            success_rate = (data['successes'] / data['count'] * 100) if data['count'] > 0 else 0
            
            stats[obj_type] = {
                'total_requests': data['count'],
                'successful_requests': data['successes'],
                'failed_requests': data['failures'],
                'success_rate_percent': round(success_rate, 2),
                'average_processing_time': round(avg_time, 3),
                'total_processing_time': round(data['total_time'], 3)
            }
        return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self.session_start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times = []
        self.object_type_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'successes': 0, 'failures': 0})

# Global monitoring instance
monitoring = PerformanceMonitoring()

class PerformanceMetrics(Resource):
    """Get current performance metrics"""
    
    def get(self):
        """
        Get performance metrics
        ---
        tags:
          - Monitoring
        summary: Get current performance metrics
        description: Returns current performance metrics including processing times, success rates, and system statistics.
        responses:
          200:
            description: Performance metrics retrieved successfully
            schema:
              type: object
              properties:
                uptime_seconds:
                  type: number
                  description: System uptime in seconds
                total_requests:
                  type: integer
                  description: Total number of requests processed
                successful_requests:
                  type: integer
                  description: Number of successful requests
                failed_requests:
                  type: integer
                  description: Number of failed requests
                success_rate_percent:
                  type: number
                  description: Success rate percentage
                average_processing_time:
                  type: number
                  description: Average processing time in seconds
                median_processing_time:
                  type: number
                  description: Median processing time in seconds
                min_processing_time:
                  type: number
                  description: Minimum processing time in seconds
                max_processing_time:
                  type: number
                  description: Maximum processing time in seconds
                requests_per_minute:
                  type: number
                  description: Requests processed per minute
        """
        try:
            metrics = monitoring.get_metrics()
            return metrics, 200
        except Exception as e:
            return make_response(jsonify({
                'error': f'Failed to get performance metrics: {str(e)}'
            }), 500)

class ObjectTypeStats(Resource):
    """Get statistics by object type"""
    
    def get(self):
        """
        Get object type statistics
        ---
        tags:
          - Monitoring
        summary: Get performance statistics by object type
        description: Returns performance statistics broken down by object type.
        responses:
          200:
            description: Object type statistics retrieved successfully
        """
        try:
            stats = monitoring.get_object_type_stats()
            return stats, 200
        except Exception as e:
            return make_response(jsonify({
                'error': f'Failed to get object type statistics: {str(e)}'
            }), 500)

class DatabaseStats(Resource):
    """Get database statistics"""
    
    def get(self):
        """
        Get database statistics
        ---
        tags:
          - Monitoring
        summary: Get database statistics
        description: Returns statistics about stored data in the database.
        responses:
          200:
            description: Database statistics retrieved successfully
        """
        try:
            # Get counts from database
            total_inputs = len(database.all(Input)) if database.all(Input) else 0
            total_outputs = len(database.all(Output)) if database.all(Output) else 0
            total_object_types = len(database.all(ObjectType)) if database.all(ObjectType) else 0
            
            # Get recent activity (last 24 hours)
            recent_outputs = []
            if database.all(Output):
                for output in database.all(Output):
                    if hasattr(output, 'created_at'):
                        # Simple check for recent activity (within last 24 hours)
                        recent_outputs.append(output)
            
            # Calculate average confidence
            confidences = []
            if database.all(Output):
                for output in database.all(Output):
                    if hasattr(output, 'pred_confidence'):
                        confidences.append(output.pred_confidence)
            
            avg_confidence = statistics.mean(confidences) if confidences else 0
            
            stats = {
                'total_inputs': total_inputs,
                'total_outputs': total_outputs,
                'total_object_types': total_object_types,
                'recent_activity_24h': len(recent_outputs),
                'average_confidence': round(avg_confidence, 3),
                'database_health': 'healthy' if total_outputs > 0 else 'empty'
            }
            
            return stats, 200
        except Exception as e:
            return make_response(jsonify({
                'error': f'Failed to get database statistics: {str(e)}'
            }), 500)

class ResetStats(Resource):
    """Reset performance statistics"""
    
    def post(self):
        """
        Reset performance statistics
        ---
        tags:
          - Monitoring
        summary: Reset all performance statistics
        description: Resets all performance monitoring statistics to zero.
        responses:
          200:
            description: Statistics reset successfully
        """
        try:
            monitoring.reset_stats()
            return {'message': 'Performance statistics reset successfully'}, 200
        except Exception as e:
            return make_response(jsonify({
                'error': f'Failed to reset statistics: {str(e)}'
            }), 500)

class SystemHealth(Resource):
    """Get comprehensive system health"""
    
    def get(self):
        """
        Get system health
        ---
        tags:
          - Monitoring
        summary: Get comprehensive system health
        description: Returns comprehensive system health including performance metrics, database stats, and AI pipeline status.
        responses:
          200:
            description: System health retrieved successfully
        """
        try:
            from ...pipeline.pipeline import pipeline
            
            # Get performance metrics
            perf_metrics = monitoring.get_metrics()
            
            # Get database stats
            total_outputs = len(database.all(Output)) if database.all(Output) else 0
            
            # Get AI pipeline status
            pipeline_status = pipeline.get_model_status()
            
            # Calculate overall health score
            health_score = 100
            if perf_metrics['success_rate_percent'] < 90:
                health_score -= 20
            if perf_metrics['average_processing_time'] > 30:
                health_score -= 15
            if not pipeline_status.get('models_loaded', False):
                health_score -= 30
            if total_outputs == 0:
                health_score -= 10
            
            health_status = 'excellent' if health_score >= 90 else 'good' if health_score >= 70 else 'fair' if health_score >= 50 else 'poor'
            
            health_data = {
                'overall_health_score': max(0, health_score),
                'health_status': health_status,
                'performance_metrics': perf_metrics,
                'pipeline_status': pipeline_status,
                'database_records': total_outputs,
                'timestamp': datetime.now().isoformat(),
                'recommendations': []
            }
            
            # Add recommendations
            if perf_metrics['success_rate_percent'] < 90:
                health_data['recommendations'].append('Consider investigating failed requests')
            if perf_metrics['average_processing_time'] > 30:
                health_data['recommendations'].append('Processing time is high, consider optimization')
            if not pipeline_status.get('models_loaded', False):
                health_data['recommendations'].append('AI models are not loaded properly')
            
            return health_data, 200
        except Exception as e:
            return make_response(jsonify({
                'error': f'Failed to get system health: {str(e)}'
            }), 500)








