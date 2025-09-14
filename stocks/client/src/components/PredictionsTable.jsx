import React from 'react';
import { Table, TableHead, TableRow, TableCell, TableBody } from '@mui/material';

export default function PredictionsTable({ rows, label = 'Model' }) {
  if (!rows || rows.length === 0) return <div style={{ opacity: 0.7 }}>No rows.</div>;
  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Date</TableCell>
          <TableCell align="right">{label} Close</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {rows.map((r, i) => (
          <TableRow key={`${r.date}-${i}`}>
            <TableCell>{String(r.date).slice(0, 10)}</TableCell>
            <TableCell align="right">{Number(r.value).toFixed(2)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
