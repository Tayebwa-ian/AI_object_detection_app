import React, { useState } from 'react';
import { Container, Typography, List, ListItem, ListItemText, Divider } from '@mui/material';
import ObjectForm from '../components/ObjectForm';
import { sendJSON } from '../CRUD/postUpdateDelete';

export default function ObjectTypePage({ initial = [] }) {
  const [objects, setObjects] = useState(initial);

  const handleCreate = async (obj) => {
    try {
      const res = await sendJSON('/api/objects', obj, 'POST');
      // assume response contains new object
      const newObj = res.data;
      setObjects((prev) => [newObj || obj, ...prev]);
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <Container sx={{ py: 4 }}>
      <Typography variant="h5" gutterBottom>Manage Object Types</Typography>
      <ObjectForm onCreate={handleCreate} />

      <List>
        {objects.map((o) => (
          <React.Fragment key={o.name || o.id}>
            <ListItem>
              <ListItemText primary={o.name} secondary={o.description} />
            </ListItem>
            <Divider component="li" />
          </React.Fragment>
        ))}
      </List>
    </Container>
  );
}
