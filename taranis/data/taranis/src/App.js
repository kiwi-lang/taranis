import './App.css';

function Sidebar() {
  return (
  <div className="sidebar light-grey bar-block">
    <a href="#" className="bar-item button">L</a>
    <a href="#" className="bar-item button">L</a>
    <a href="#" className="bar-item button">L</a>
  </div>
  )
}

function App() {
  return (
    <div className="App">
      <header className="App-header">
          {Sidebar()}
          <div className="main" style={{marginLeft:(64 + 8) + 'px'}}>
            <div className="container">
              <h1>Hello</h1>
            </div>
        </div>
      </header>
    </div>
  );
}

export default App;
