import './App.css';

import React, { useState, useEffect } from 'react';
import { Routes, Route, Outlet, Link } from "react-router-dom";
import { BrowserRouter } from "react-router-dom";
import { VegaLite } from 'react-vega';


const API_URL = "http://127.0.0.1:8000"

function Request(path) {
 return fetch(`${API_URL}/${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  })
} 


function GetSpec(data, width, height) {
  const dataspec = {
      "values": data
  }
  return {
    "config": {
      "view": {
        "continuousWidth": width,
        "continuousHeight": height
      }
    }, 
    "layer": [
      {
        "data": dataspec,
        "mark": "line",
        "encoding": {
          "color": {
            "field": "name",
            "type": "nominal"
          },
          "x": {
            "field": "time",
            "scale": {
              "type": "log"
            },
            "type": "quantitative"
          },
          "y": {
            "aggregate": "min",
            "field": "value",
            "type": "quantitative"
          }
        },
        "transform": [
          {
            "filter": "(datum.metric === 'train_loss')"
          }
        ]
      },
      {
        "data": dataspec,
        "mark": {
          "type": "line",
          "strokeDash": [
            1,
            1
          ]
        },
        "encoding": {
          "color": {
            "field": "name",
            "type": "nominal"
          },
          "x": {
            "field": "time",
            "scale": {
              "type": "log"
            },
            "type": "quantitative"
          },
          "y": {
            "aggregate": "mean",
            "field": "value",
            "type": "quantitative"
          }
        },
        "transform": [
          {
            "filter": "(datum.metric === 'train_loss')"
          }
        ]
      }
    ],
    "$schema": "https://vega.github.io/schema/vega-lite/v5.15.0.json"
  }
}
 
function Table({Log}) {
  const [rows, _setRows] = useState([])
  const [page, setPage] = useState(1)
  const [pageItemCount, _setPageItemCount] = useState(10)
  const [pageCount, _setPageCount] = useState(1)
  const [IsLoading, _setIsLoading] = useState(false)

  function setRows(newRows) {
    _setRows(newRows)
    _setPageCount(newRows.length / pageItemCount)
  }

  function setPageItemCount(count) {
    _setPageItemCount(count)
    _setPageCount(rows.length / count)
  }


  function getPage() {
    return rows.slice((page - 1) * pageItemCount, page * pageItemCount)
  }

  let updateMessage = 'NILL'

  useEffect(() => {
    const start = new Date().getTime();

    if (updateMessage == 'NILL'){
      updateMessage = setInterval(function() {
        const now = new Date().getTime();
        const elapsed = (now - start) / 1000;
        Log("Loading data...   (" + elapsed + " sec)")
      }, 500)
    }

    Request('group/fetch/metric/1').then((response) => {
      response.json().then(data => {
        setRows(data);
        clearInterval(updateMessage)
        updateMessage = 'DONE'

        const now = new Date().getTime();
        const elapsed = (now - start) / 1000;
        Log("Finished in " + elapsed + ' sec')

        setTimeout(() => { Log("") }, 5000)
      });
    })
  }, [])


  function Header({columns}) {
    return (
        <thead>
          <tr>
            {columns.map((column) => (<th key={column}>{column}</th>))}
          </tr>
        </thead>
      )
  }

  function Row({row, columns}) {
    return (
      <tr>
        {columns.map((column) => (<td key={column}>{row[column]}</td>))}
      </tr>
    );
  }
  
  if (rows.length > 0) {
    const columns = Object.keys(rows[0]);
    const selection = getPage()

    return ( 
      <div>
        <VegaLite spec={GetSpec(rows, 1200, 600)} />
      
        <table className="table">
          {<Header columns={columns}/>}
          <tbody>
            {selection.map((row, index) => <Row key={index} row={row} columns={columns} />)}
          </tbody>
        </table>
      </div>
    )
  }
  return (
    <table></table>
  )
}

function Root({Log}) {
  return (
    <div>
      <h1>Hello</h1>
      {Table({Log})}
    </div>
  )
}

function Layout({notficationMessage}) {
  return (
    <div>
      <div className="sidebar light-grey bar-block">
        <Link className="bar-item button" to="/">H</Link>
        <Link className="bar-item button" to="/about">A</Link>
        <Link className="bar-item button" to="/about">B</Link>
        <Link className="bar-item button" to="/about">C</Link>
        <Link className="bar-item button" to="/about">D</Link>
        <Link className="bar-item button" to="/about">E</Link>
        <Link className="bar-item button" to="/about">F</Link>
        <Link className="bar-item button" to="/about">G</Link>
        <Link className="bar-item button" to="/about">H</Link>
        <Link className="bar-item button" to="/about">I</Link>
        <Link className="bar-item button" to="/about">J</Link>
        <Link className="bar-item button" to="/about">K</Link>
        <Link className="bar-item button" to="/about">L</Link>
        <Link className="bar-item button" to="/about">M</Link>
        <Link className="bar-item button" to="/about">N</Link>
        <Link className="bar-item button" to="/about">O</Link>
      </div>
      <div className="main" style={{marginLeft:(64 + 8) + 'px'}}>
        <div className="container">
          <Outlet />
        </div>
        <div className="notification-bottom">
          <div className="notification-msg">
            <p>{notficationMessage}</p>
          </div>
        </div>
      </div>
    </div>
  )
} 


function App() {
  const [notficationMessage, setNotificationMessage] = useState("")

  return (
      <div className="App">
        <header className="App-header">
        </header> 
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout notficationMessage={notficationMessage}/>}>
              <Route index element={<Root Log={setNotificationMessage}/>} />
              <Route path="about" element={<Root Log={setNotificationMessage}/>} />
            </Route>
          </Routes>
        </BrowserRouter>
      </div>
  );
}

export default App;
