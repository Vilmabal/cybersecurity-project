from unittest.mock import Mock, patch

from src.features import (
    suspicious_keyword_count,
    domain_age,
    dns_record_exists,
    number_of_external_links,
    number_of_password_fields,
    url_path_entropy,
)


# Tests suspicious keywords in the domain
def test_suspicious_keyword_count():
    assert suspicious_keyword_count(
        "https://login.example.com"
    ) == 1

    assert suspicious_keyword_count(
        "https://login.security.example.com"
    ) == 2

    assert suspicious_keyword_count(
        "https://example.com"
    ) == 0


# Tests domain age when RDAP returns a registration date
@patch("src.features.requests.get")
def test_domain_age(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "events": [
            {
                "eventAction": "registration",
                "eventDate": "2020-01-01T00:00:00Z"
            }
        ]
    }

    mock_get.return_value = mock_response

    age = domain_age("https://example.com")

    assert age > 0


# Tests domain age when RDAP cannot find the domain
@patch("src.features.requests.get")
def test_domain_age_not_found(mock_get):
    mock_response = Mock()
    mock_response.status_code = 404

    mock_get.return_value = mock_response

    assert domain_age("https://example.com") == -1


# Tests DNS when the domain resolves successfully
@patch("src.features.socket.gethostbyname")
def test_dns_record_exists(mock_gethostbyname):
    mock_gethostbyname.return_value = "93.184.216.34"

    assert dns_record_exists("https://example.com") is True


# Tests DNS when the domain does not resolve
@patch("src.features.socket.gethostbyname")
def test_dns_record_does_not_exist(mock_gethostbyname):
    import socket

    mock_gethostbyname.side_effect = socket.gaierror

    assert dns_record_exists("https://does-not-exist.example") is False


# Tests counting external links
@patch("src.features.requests.get")
def test_number_of_external_links(mock_get):
    mock_response = Mock()
    mock_response.text = """
        <html>
            <a href="https://example.com/page">Internal</a>
            <a href="https://google.com">External</a>
            <a href="https://github.com">External</a>
        </html>
    """

    mock_get.return_value = mock_response

    assert number_of_external_links(
        "https://example.com"
    ) == 2


# Tests pages without external links
@patch("src.features.requests.get")
def test_number_of_external_links_none(mock_get):
    mock_response = Mock()
    mock_response.text = """
        <html>
            <a href="https://example.com/page">Internal</a>
        </html>
    """

    mock_get.return_value = mock_response

    assert number_of_external_links(
        "https://example.com"
    ) == 0


# Tests counting password fields
@patch("src.features.requests.get")
def test_number_of_password_fields(mock_get):
    mock_response = Mock()
    mock_response.text = """
        <html>
            <input type="text">
            <input type="password">
            <input type="password">
        </html>
    """

    mock_get.return_value = mock_response

    assert number_of_password_fields(
        "https://example.com"
    ) == 2


# Tests page without password fields
@patch("src.features.requests.get")
def test_number_of_password_fields_none(mock_get):
    mock_response = Mock()
    mock_response.text = """
        <html>
            <input type="text">
            <input type="email">
        </html>
    """

    mock_get.return_value = mock_response

    assert number_of_password_fields(
        "https://example.com"
    ) == 0


# Tests empty URL path
def test_url_path_entropy_empty():
    assert url_path_entropy("https://example.com") == 0.0


# Tests a normal URL path
def test_url_path_entropy_normal():
    entropy = url_path_entropy(
        "https://example.com/hello-world"
    )

    assert entropy > 0


# Tests a random-looking URL path
def test_url_path_entropy_random():
    entropy = url_path_entropy(
        "https://example.com/e0jhUthcl9"
    )

    assert entropy > 0