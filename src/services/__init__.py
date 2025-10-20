"""
Services Module - RAG, Search, Classification, Booking, Document Processing
"""

from .rag import search_medical_info
from .search import brave_search
from .classification import classify_image
from .appointment_booking import extract_information, update_appointment_info, appointment_info
from .documents import (
    process_uploaded_pdf,
    search_session_documents,
    list_user_sessions,
    delete_session_documents,
    get_session_stats
)

__all__ = [
    'search_medical_info',
    'brave_search', 
    'classify_image',
    'extract_information',
    'update_appointment_info',
    'appointment_info',
    'process_uploaded_pdf',
    'search_session_documents',
    'list_user_sessions',
    'delete_session_documents',
    'get_session_stats'
]
