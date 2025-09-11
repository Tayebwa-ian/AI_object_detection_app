import { useState, useEffect } from 'react';
import { getJSON } from '../CRUD/get';

export const useGet = (url) => {
  const [data, setData] = useState(null);
  const [isPending, setIsPending] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    setIsPending(true);
    setError(null);

    getJSON(url).then(({ data }) => {
      if (!mounted) return;
      setData(data);
      setIsPending(false);
    }).catch((err) => {
      if (!mounted) return;
      setError(err.message || String(err));
      setIsPending(false);
    });

    return () => { mounted = false; };
  }, [url]);

  return { data, isPending, error };
};
