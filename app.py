/* Main styles */
.main {
    padding: 2rem;
}

/* Headers */
h1 {
    color: #0E86D4;
    font-weight: 700;
    margin-bottom: 1rem;
}

h2 {
    color: #262730;
    font-weight: 600;
    margin-top: 2rem;
}

h3 {
    color: #0E86D4;
    font-weight: 500;
    padding: 0.5rem 0;
}

/* Cards */
div[data-testid="stVerticalBlock"] > div {
    background: white;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #F8F9FA;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #E9ECEF;
}

div[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0E86D4;
}

/* Tables */
div[data-testid="stTable"] {
    border: 1px solid #E9ECEF;
    border-radius: 0.5rem;
    overflow: hidden;
}

div[data-testid="stTable"] table {
    border-collapse: collapse;
    width: 100%;
}

div[data-testid="stTable"] th {
    background: #F8F9FA;
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
}

div[data-testid="stTable"] td {
    padding: 0.75rem;
    border-top: 1px solid #E9ECEF;
}

/* Tabs */
div[data-testid="stTabs"] button {
    font-weight: 500;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0E86D4;
    border-bottom-color: #0E86D4;
}

/* Sentiment colors */
.positive {
    color: #28A745;
}

.negative {
    color: #DC3545;
}

.neutral {
    color: #6C757D;
}

/* Company symbols */
code {
    background: #F8F9FA;
    padding: 0.2rem 0.4rem;
    border-radius: 0.25rem;
    color: #0E86D4;
    font-size: 0.9em;
}

/* Blockquotes for article descriptions */
blockquote {
    border-left: 4px solid #0E86D4;
    margin: 1rem 0;
    padding: 0.5rem 1rem;
    background: #F8F9FA;
    color: #495057;
}

/* Success messages */
div[data-testid="stSuccessMessage"] {
    background: #D4EDDA;
    border-color: #C3E6CB;
    color: #155724;
}

/* Info messages */
div[data-testid="stInfoMessage"] {
    background: #CCE5FF;
    border-color: #B8DAFF;
    color: #004085;
}

/* Warning messages */
div[data-testid="stWarningMessage"] {
    background: #FFF3CD;
    border-color: #FFEEBA;
    color: #856404;
}

/* Error messages */
div[data-testid="stErrorMessage"] {
    background: #F8D7DA;
    border-color: #F5C6CB;
    color: #721C24;
}

/* Links */
a {
    color: #0E86D4;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Sidebar */
.sidebar .sidebar-content {
    background: #FFFFFF;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    border: 2px dashed #DEE2E6;
    border-radius: 0.5rem;
    padding: 1rem;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #0E86D4;
}
