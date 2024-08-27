import {Route, Routes} from 'react-router-dom';
import './App.css';
import WebCamPage from './pages/WebCamPage';

function App() {
  return (
    <>
      <Routes>
        <Route path="/webcam" element={<WebCamPage/>}/>
      </Routes>
    </>
  );
}

export default App;
