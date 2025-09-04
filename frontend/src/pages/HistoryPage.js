import React from 'react';
import { Container, Typography } from '@mui/material';
import { useGet } from '../hooks/useGet';
import DataLoading from '../components/DataLoading';
import ErrorDisplay from '../components/ErrorDisplay';
import HorizontalCard from '../components/HorizontalCard';

export default function HistoryPage() {
  const { data, isPending, error } = useGet('/api/history');
  if (isPending) return <DataLoading />;
  if (error) return <ErrorDisplay error={error} />;

  const items = (data && data.items) || [];

  return (
    <Container sx={{ py: 4 }}>
      <Typography variant="h5" gutterBottom>Prediction History</Typography>
      {items.length === 0 && <Typography variant="body2">No history yet</Typography>}
      {items.map((it) => (
        <HorizontalCard key={it.id} data={it} />
      ))}
    </Container>
  );
}
