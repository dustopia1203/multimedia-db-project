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
  Accordion,
  Table,
  Image,
  Tabs,
  Tab,
} from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

function App() {
  const [audioFiles, setAudioFiles] = useState([]);
  const [uploadedFileMetadata, setUploadedFileMetadata] = useState(null);
  const [uploadedFileUrl, setUploadedFileUrl] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setError("");
    setUploadedFileMetadata(null);
    setAudioFiles([]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Vui lòng chọn một tệp âm thanh để tải lên.");
      return;
    }

    setLoading(true);
    setError("");
    setUploadedFileMetadata(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Upload file to get metadata
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Đã xảy ra lỗi khi tải lên tệp âm thanh.");
      }

      const data = await response.json();
      setUploadedFileMetadata(data);
      setUploadedFileUrl(URL.createObjectURL(selectedFile));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!selectedFile) {
      setError("Vui lòng chọn một tệp âm thanh để tìm kiếm.");
      return;
    }

    setSearching(true);
    setError("");
    setAudioFiles([]);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/upload/q", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Đã xảy ra lỗi khi tìm kiếm âm thanh tương tự.");
      }

      const data = await response.json();
      setAudioFiles(data);

      if (!uploadedFileUrl) {
        setUploadedFileUrl(URL.createObjectURL(selectedFile));
      }

      if (data.length === 0) {
        setError("Không tìm thấy kết quả nào tương tự.");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  };

  // Helper function to format feature values
  const formatFeatureValue = (value) => {
    if (typeof value === "number") {
      return value.toFixed(4);
    }
    return value;
  };

  // Helper function to render statistical features
  const renderStatFeature = (feature) => {
    if (!feature) return null;

    return (
      <div className="feature-stats">
        <small>Trung bình: {formatFeatureValue(feature.mean)}</small>
      </div>
    );
  };

  // Function to render file metadata
  const renderFileMetadata = (file) => {
    if (!file) return null;

    return (
      <Card className="shadow-sm mb-4" style={{ borderRadius: "10px" }}>
        <Card.Body>
          <Card.Title>{file.filename}</Card.Title>
          <div className="d-flex justify-content-between mb-3">
            <div>
              <Card.Text>Thời lượng: {file.duration.toFixed(2)} giây</Card.Text>
              <Card.Text>Tần số mẫu: {file.sample_rate} Hz</Card.Text>
            </div>
            <div>
              <audio controls className="mt-2">
                <source src={uploadedFileUrl} type="audio/mpeg" />
                Trình duyệt của bạn không hỗ trợ phát âm thanh.
              </audio>
            </div>
          </div>

          <Accordion className="mt-3">
            <Accordion.Item eventKey="0">
              <Accordion.Header>Đặc trưng âm thanh</Accordion.Header>
              <Accordion.Body>
                {file.spectrogram && (
                  <div className="mb-3">
                    <h6>Phổ tần số</h6>
                    <div className="text-center">
                      <Image
                        src={`data:image/png;base64,${file.spectrogram}`}
                        fluid
                        alt="Spectrogram"
                        style={{ maxHeight: "200px" }}
                      />
                    </div>
                  </div>
                )}

                <h6>Thông số cơ bản</h6>
                <Table size="sm" bordered hover>
                  <tbody>
                    <tr>
                      <td width="40%">Tỉ lệ cắt không (ZCR)</td>
                      <td>
                        {renderStatFeature(
                          file.feature_details.zero_crossing_rate
                        )}
                      </td>
                    </tr>
                    <tr>
                      <td>Năng lượng RMS</td>
                      <td>
                        {renderStatFeature(file.feature_details.rms_energy)}
                      </td>
                    </tr>
                    <tr>
                      <td>Trọng tâm phổ</td>
                      <td>
                        {renderStatFeature(
                          file.feature_details.spectral_centroid
                        )}
                      </td>
                    </tr>
                    <tr>
                      <td>Độ lăn phổ</td>
                      <td>
                        {renderStatFeature(
                          file.feature_details.spectral_rolloff
                        )}
                      </td>
                    </tr>
                    <tr>
                      <td>Thay đổi phổ</td>
                      <td>
                        {renderStatFeature(file.feature_details.spectral_flux)}
                      </td>
                    </tr>
                  </tbody>
                </Table>

                <h6>Đặc trưng MFCC (trung bình)</h6>
                <div className="feature-array">
                  {file.feature_details.mfcc_mean?.map((val, i) => (
                    <div key={i} className="feature-item">
                      <small>
                        {i + 1}: {val.toFixed(3)}
                      </small>
                    </div>
                  ))}
                </div>

                <h6>Đặc trưng Chroma (trung bình)</h6>
                <div className="feature-array">
                  {file.feature_details.chroma_mean?.map((val, i) => (
                    <div key={i} className="feature-item">
                      <small>
                        {
                          [
                            "C",
                            "C#",
                            "D",
                            "D#",
                            "E",
                            "F",
                            "F#",
                            "G",
                            "G#",
                            "A",
                            "A#",
                            "B",
                          ][i]
                        }
                        : {val.toFixed(3)}
                      </small>
                    </div>
                  ))}
                </div>

                <h6>Độ tương phản phổ</h6>
                <div className="feature-array">
                  {file.feature_details.spectral_contrast?.map((val, i) => (
                    <div key={i} className="feature-item">
                      <small>
                        Băng {i + 1}: {val.toFixed(3)}
                      </small>
                    </div>
                  ))}
                </div>
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </Card.Body>
      </Card>
    );
  };

  return (
    <>
      <Container className="mt-5">
        <h1 className="text-center mb-4">Tra cứu âm thanh nhạc cụ khí</h1>
        <Form className="d-flex justify-content-center">
          <Row className="mb-4 w-100">
            <Col md={8}>
              <Form.Control
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
              />
            </Col>
            <Col md={4} className="d-flex gap-2">
              <Button
                variant="secondary"
                onClick={handleUpload}
                disabled={loading || !selectedFile}
                className="flex-grow-1"
              >
                {loading ? (
                  <Spinner animation="border" size="sm" />
                ) : (
                  "Phân tích"
                )}
              </Button>
              <Button
                variant="primary"
                onClick={handleSearch}
                disabled={searching || !selectedFile}
                className="flex-grow-1"
              >
                {searching ? (
                  <Spinner animation="border" size="sm" />
                ) : (
                  "Tìm kiếm"
                )}
              </Button>
            </Col>
          </Row>
        </Form>

        {error && <p className="text-danger text-center mb-4">{error}</p>}

        {uploadedFileMetadata && (
          <div className="mb-4">
            <h3 className="text-center mb-3">
              Thông tin tệp âm thanh đã tải lên
            </h3>
            {renderFileMetadata(uploadedFileMetadata)}
          </div>
        )}

        {audioFiles.length > 0 && (
          <div className="mb-4">
            <h3 className="text-center mb-3">Kết quả tìm kiếm tương tự</h3>
            <Row>
              {audioFiles.map((file, index) => (
                <Col md={12} key={index} className="mb-4">
                  <Card className="shadow-sm" style={{ borderRadius: "10px" }}>
                    <Card.Body>
                      <Card.Title>{file.filename}</Card.Title>
                      <div className="d-flex justify-content-between mb-3">
                        <div>
                          <Card.Text>
                            Thời lượng: {file.duration.toFixed(2)} giây
                          </Card.Text>
                          <Card.Text>
                            Tần số mẫu: {file.sample_rate} Hz
                          </Card.Text>
                          <Card.Text>
                            <strong>
                              Độ tương đồng: {file.similarity_score}
                            </strong>
                          </Card.Text>
                        </div>
                        <div>
                          <audio controls className="mt-2">
                            <source
                              src={`http://localhost:8000/training-data/${file.filename}`}
                              type="audio/mpeg"
                            />
                            Trình duyệt của bạn không hỗ trợ phát âm thanh.
                          </audio>
                        </div>
                      </div>

                      <Accordion className="mt-3">
                        <Accordion.Item eventKey="0">
                          <Accordion.Header>
                            Đặc trưng âm thanh
                          </Accordion.Header>
                          <Accordion.Body>
                            {file.spectrogram && (
                              <div className="mb-3">
                                <h6>Phổ tần số</h6>
                                <div className="text-center">
                                  <Image
                                    src={`data:image/png;base64,${file.spectrogram}`}
                                    fluid
                                    alt="Spectrogram"
                                    style={{ maxHeight: "200px" }}
                                  />
                                </div>
                              </div>
                            )}

                            <h6>Thông số cơ bản</h6>
                            <Table size="sm" bordered hover>
                              <tbody>
                                <tr>
                                  <td width="40%">Tỉ lệ cắt không (ZCR)</td>
                                  <td>
                                    {renderStatFeature(
                                      file.feature_details.zero_crossing_rate
                                    )}
                                  </td>
                                </tr>
                                <tr>
                                  <td>Năng lượng RMS</td>
                                  <td>
                                    {renderStatFeature(
                                      file.feature_details.rms_energy
                                    )}
                                  </td>
                                </tr>
                                <tr>
                                  <td>Trọng tâm phổ</td>
                                  <td>
                                    {renderStatFeature(
                                      file.feature_details.spectral_centroid
                                    )}
                                  </td>
                                </tr>
                                <tr>
                                  <td>Độ lăn phổ</td>
                                  <td>
                                    {renderStatFeature(
                                      file.feature_details.spectral_rolloff
                                    )}
                                  </td>
                                </tr>
                                <tr>
                                  <td>Thay đổi phổ</td>
                                  <td>
                                    {renderStatFeature(
                                      file.feature_details.spectral_flux
                                    )}
                                  </td>
                                </tr>
                              </tbody>
                            </Table>

                            <h6>Đặc trưng MFCC (trung bình)</h6>
                            <div className="feature-array">
                              {file.feature_details.mfcc_mean?.map((val, i) => (
                                <div key={i} className="feature-item">
                                  <small>
                                    {i + 1}: {val.toFixed(3)}
                                  </small>
                                </div>
                              ))}
                            </div>

                            <h6>Đặc trưng Chroma (trung bình)</h6>
                            <div className="feature-array">
                              {file.feature_details.chroma_mean?.map(
                                (val, i) => (
                                  <div key={i} className="feature-item">
                                    <small>
                                      {
                                        [
                                          "C",
                                          "C#",
                                          "D",
                                          "D#",
                                          "E",
                                          "F",
                                          "F#",
                                          "G",
                                          "G#",
                                          "A",
                                          "A#",
                                          "B",
                                        ][i]
                                      }
                                      : {val.toFixed(3)}
                                    </small>
                                  </div>
                                )
                              )}
                            </div>

                            <h6>Độ tương phản phổ</h6>
                            <div className="feature-array">
                              {file.feature_details.spectral_contrast?.map(
                                (val, i) => (
                                  <div key={i} className="feature-item">
                                    <small>
                                      Băng {i + 1}: {val.toFixed(3)}
                                    </small>
                                  </div>
                                )
                              )}
                            </div>
                          </Accordion.Body>
                        </Accordion.Item>
                      </Accordion>
                    </Card.Body>
                  </Card>
                </Col>
              ))}
            </Row>
          </div>
        )}
      </Container>
    </>
  );
}

export default App;
