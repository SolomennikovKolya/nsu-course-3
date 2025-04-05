import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('Loading...');
  const [counter, setCounter] = useState(0);

  useEffect(() => {
    fetch('/api/hello')
      .then(res => res.json())
      .then(data => setMessage(data.message))
      .catch(err => setMessage('Error fetching data'));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Flask + React App</h1>
        <p>Message from Flask: {message}</p>
        <div>
          <p>Counter: {counter}</p>
          <button onClick={() => setCounter(counter + 1)}>
            Increment
          </button>
        </div>
      </header>
    </div>
  );
}

export default App;
