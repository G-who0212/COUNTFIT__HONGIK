import React,{ useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import {styled} from 'styled-components';


function WebCamPage(props) {
    const webcamRef = useRef(null);
    // const [imgSrc, setImgSrc] = useState(null);

    // const capture = useCallback(() => {
    //     const imageSrc = webcamRef.current.getScreenshot();
    //     setImgSrc(imageSrc);
    // }, [webcamRef, setImgSrc]);

    return (
        <div>
            <WebCamContainer>
            <Webcam
                audio={false}
                ref={webcamRef}
                // screenshotFormat="image/jpeg"
                //heght={1000}
                width={1000}
            />
            </WebCamContainer>
            {/* <button onClick={capture}>Capture photo</button>
            {imgSrc && (
                <img
                     src={imgSrc}
                />
            )} */}
        </div>
    );
}

const WebCamContainer = styled.div`
    height: 100vh;
    display: flex;
    
`

export default WebCamPage;