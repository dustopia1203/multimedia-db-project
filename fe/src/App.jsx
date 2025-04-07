import React, { useState } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Button,
  Form,
  InputGroup,
  Spinner,
} from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

function App() {
  const [audioFiles, setAudioFiles] = useState([]);
  const [uploadedFileUrl, setUploadedFileUrl] = useState(null); // Store the uploaded file URL
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setError(""); // Clear any previous errors
  };

  const handleSearch = async () => {
    if (!selectedFile) {
      setError("Vui lòng chọn một tệp âm thanh để tìm kiếm.");
      return;
    }

    setLoading(true);
    setError("");
    setAudioFiles([]);
    setUploadedFileUrl(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/upload/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Đã xảy ra lỗi khi tải lên tệp âm thanh.");
      }

      const data = await response.json();
      setUploadedFileUrl(URL.createObjectURL(selectedFile)); // Create a URL for the uploaded file
      setAudioFiles(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Container className="mt-5">
        <h1 className="text-center mb-4">Tra cứu âm thanh nhạc cụ khí</h1>
        <Form className="d-flex justify-content-center">
          <Row className="mb-4">
            <Col md={8}>
              <Form.Control
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
              />
            </Col>
            <Col md={4}>
              <InputGroup>
                <Button
                  variant="primary"
                  onClick={handleSearch}
                  disabled={loading}
                >
                  {loading ? (
                    <Spinner animation="border" size="sm" />
                  ) : (
                    "Tìm kiếm"
                  )}
                </Button>
              </InputGroup>
            </Col>
          </Row>
        </Form>
        {error && <p className="text-danger text-center">{error}</p>}
        {uploadedFileUrl && (
          <div className="text-center mb-4">
            <h5>File đã tải lên:</h5>
            <audio controls>
              <source src={uploadedFileUrl} type="audio/mpeg" />
              Trình duyệt của bạn không hỗ trợ phát âm thanh.
            </audio>
          </div>
        )}
        <Row>
          {audioFiles.map((file, index) => (
            <Col md={4} key={index} className="mb-4">
              <Card
                className="h-100 shadow-sm"
                style={{ borderRadius: "10px" }}
              >
                <Card.Body>
                  <Card.Title>{file.filename}</Card.Title>
                  <Card.Text>
                    Thời lượng: {file.duration.toFixed(2)} giây
                  </Card.Text>
                  <Card.Text>Tần số mẫu: {file.sample_rate} Hz</Card.Text>
                  <Card.Text>
                    Điểm tương đồng: {file.similarity_score}
                  </Card.Text>
                  <audio controls>
                    <source
                      src={`http://localhost:8000/training-data/${file.filename}`}
                      type="audio/mpeg"
                    />
                    Trình duyệt của bạn không hỗ trợ phát âm thanh.
                  </audio>
                </Card.Body>
              </Card>
            </Col>
          ))}
        </Row>
      </Container>
    </>
  );
}

export default App;
