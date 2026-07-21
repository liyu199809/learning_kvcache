import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Text, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime, date, time
import os

DATABASE_URL = os.getenv("DATABASE_PATH", "sqlite:///outputs/databases/volunteermatch.db")

engine = create_engine('sqlite:///outputs/servers/20260706_162049_marketplace_1/final.db', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="VolunteerMatch Simplified API", version="1.0.0")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    full_name = Column(String)
    location_city = Column(String)
    location_state = Column(String)
    location_zip = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Skill(Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class UserSkill(Base):
    __tablename__ = "user_skills"
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    skill_id = Column(Integer, ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True)
    proficiency_level = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class UserAvailability(Base):
    __tablename__ = "user_availability"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    weekday = Column(Integer, nullable=False)
    start_time = Column(String, nullable=False)
    end_time = Column(String, nullable=False)
    location_zip = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Cause(Base):
    __tablename__ = "causes"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Organization(Base):
    __tablename__ = "organizations"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    location_city = Column(String)
    location_state = Column(String)
    location_zip = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class OrganizationContact(Base):
    __tablename__ = "organization_contacts"
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    email = Column(String)
    phone = Column(String)
    role = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Opportunity(Base):
    __tablename__ = "opportunities"
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    is_virtual = Column(Integer, nullable=False, default=0)
    location_city = Column(String)
    location_state = Column(String)
    location_zip = Column(String)
    cause_id = Column(Integer, ForeignKey("causes.id", ondelete="SET NULL"))
    min_hours_per_week = Column(Float)
    max_hours_per_week = Column(Float)
    min_commitment_months = Column(Integer)
    start_date = Column(Date)
    end_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class OpportunitySkill(Base):
    __tablename__ = "opportunity_skills"
    opportunity_id = Column(Integer, ForeignKey("opportunities.id", ondelete="CASCADE"), primary_key=True)
    skill_id = Column(Integer, ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True)
    required_level = Column(String)


class OpportunityShift(Base):
    __tablename__ = "opportunity_shifts"
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(Integer, ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False)
    start_datetime = Column(DateTime, nullable=False)
    end_datetime = Column(DateTime, nullable=False)
    max_volunteers = Column(Integer)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    opportunity_id = Column(Integer, ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False)
    message = Column(Text)
    include_profile_skills = Column(Integer, nullable=False, default=0)
    include_profile_availability = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ApplicationMessage(Base):
    __tablename__ = "application_messages"
    id = Column(Integer, primary_key=True)
    application_id = Column(Integer, ForeignKey("applications.id", ondelete="CASCADE"), nullable=False)
    sender_type = Column(String, nullable=False)
    sender_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    organization_contact_id = Column(Integer, ForeignKey("organization_contacts.id", ondelete="SET NULL"))
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Roster(Base):
    __tablename__ = "rosters"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    opportunity_id = Column(Integer, ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False)
    role = Column(String)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ShiftAssignment(Base):
    __tablename__ = "shift_assignments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    opportunity_id = Column(Integer, ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False)
    shift_id = Column(Integer, ForeignKey("opportunity_shifts.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False)
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ShiftNote(Base):
    __tablename__ = "shift_notes"
    id = Column(Integer, primary_key=True)
    shift_assignment_id = Column(Integer, ForeignKey("shift_assignments.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    note = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class VolunteerHour(Base):
    __tablename__ = "volunteer_hours"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    opportunity_id = Column(Integer, ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False)
    shift_id = Column(Integer, ForeignKey("opportunity_shifts.id", ondelete="SET NULL"))
    date = Column(Date, nullable=False)
    start_datetime = Column(DateTime)
    end_datetime = Column(DateTime)
    hours = Column(Float, nullable=False)
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class OpportunityRating(Base):
    __tablename__ = "opportunity_ratings"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    opportunity_id = Column(Integer, ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer, nullable=False)
    review = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Favorite(Base):
    __tablename__ = "favorites"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    opportunity_id = Column(Integer, ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False)
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


class UserModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Current user identifier (always 1 in this context)", example=1)
    username: str = Field(..., description="Unique username", example="volunteer_jane")
    email: str = Field(..., description="User email address", example="jane@example.org")
    full_name: Optional[str] = Field(None, description="Full display name", example="Jane Doe")
    location_city: Optional[str] = Field(None, description="User city", example="San Francisco")
    location_state: Optional[str] = Field(None, description="User state or region", example="CA")
    location_zip: Optional[str] = Field(None, description="User ZIP or postal code", example="94103")
    created_at: datetime = Field(..., description="User creation timestamp (ISO 8601)", example="2025-01-10T12:00:00Z")
    updated_at: datetime = Field(..., description="Last user update timestamp (ISO 8601)", example="2025-02-01T09:30:00Z")


class UserResponse(BaseModel):
    user: UserModel = Field(..., description="User profile object", example={})


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = Field(None, description="Updated full name", example="Jane Q. Volunteer")
    location_city: Optional[str] = Field(None, description="Updated city", example="San Francisco")
    location_state: Optional[str] = Field(None, description="Updated state or region", example="CA")
    location_zip: Optional[str] = Field(None, description="Updated ZIP or postal code", example="94103")


class SkillModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Skill identifier", example=3)
    name: str = Field(..., description="Unique skill name", example="Spanish translation")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-05T10:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-05T10:00:00Z")


class SkillsListResponse(BaseModel):
    skills: List[SkillModel] = Field(..., description="List of skills", example=[])


class CreateSkillRequest(BaseModel):
    name: str = Field(..., description="Unique name of the skill to create", example="event photography")


class SkillResponse(BaseModel):
    skill: SkillModel = Field(..., description="Skill object", example={})


class UserSkillModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    skill_id: int = Field(..., description="Skill identifier", example=3)
    skill_name: str = Field(..., description="Skill name", example="Spanish translation")
    proficiency_level: Optional[str] = Field(None, description="Optional user-defined proficiency level", example="intermediate")
    created_at: datetime = Field(..., description="Timestamp when the skill was added to the user", example="2025-01-12T14:05:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp for this user skill", example="2025-01-12T14:05:00Z")


class UserSkillsListResponse(BaseModel):
    skills: List[UserSkillModel] = Field(..., description="List of user skills", example=[])


class AddUserSkillRequest(BaseModel):
    skill_id: int = Field(..., description="Identifier of the existing skill to add", example=3)
    proficiency_level: Optional[str] = Field(None, description="User-defined proficiency level label", example="advanced")


class UserSkillRecordModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_id: int = Field(..., description="User identifier", example=1)
    skill_id: int = Field(..., description="Skill identifier", example=3)
    proficiency_level: Optional[str] = Field(None, description="Stored proficiency level", example="advanced")
    created_at: datetime = Field(..., description="Timestamp when the skill was linked to user", example="2025-01-12T14:05:00Z")
    updated_at: datetime = Field(..., description="Last updated timestamp", example="2025-01-12T14:05:00Z")


class UserSkillResponse(BaseModel):
    user_skill: UserSkillRecordModel = Field(..., description="User skill record", example={})


class UpdateUserSkillRequest(BaseModel):
    proficiency_level: str = Field(..., description="New proficiency level value", example="expert")


class DeleteSuccessResponse(BaseModel):
    success: bool = Field(..., description="True if the record was deleted", example=True)


class UserAvailabilityModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Availability record identifier", example=10)
    weekday: int = Field(..., description="Weekday 0=Monday ... 6=Sunday", example=0)
    start_time: str = Field(..., description="Start time in HH:MM 24h format", example="18:00")
    end_time: str = Field(..., description="End time in HH:MM 24h format", example="21:00")
    location_zip: Optional[str] = Field(None, description="ZIP code where user is available for this slot", example="94103")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-12T15:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-12T15:00:00Z")


class UserAvailabilityListResponse(BaseModel):
    availability: List[UserAvailabilityModel] = Field(..., description="List of availability windows", example=[])


class CreateAvailabilityRequest(BaseModel):
    weekday: int = Field(..., description="Weekday number 0=Monday ... 6=Sunday", example=0)
    start_time: str = Field(..., description="Start time in HH:MM 24h format", example="18:00")
    end_time: str = Field(..., description="End time in HH:MM 24h format", example="21:00")
    location_zip: Optional[str] = Field(None, description="ZIP code for this availability block", example="94103")


class UserAvailabilityWithUserModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Availability record identifier", example=11)
    user_id: int = Field(..., description="User identifier", example=1)
    weekday: int = Field(..., description="Weekday 0=Monday ... 6=Sunday", example=2)
    start_time: str = Field(..., description="Start time", example="18:00")
    end_time: str = Field(..., description="End time", example="21:00")
    location_zip: Optional[str] = Field(None, description="ZIP code for this slot", example="94103")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-12T15:05:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-12T15:05:00Z")


class UserAvailabilityResponse(BaseModel):
    availability: UserAvailabilityWithUserModel = Field(..., description="Availability record", example={})


class UpdateAvailabilityRequest(BaseModel):
    weekday: Optional[int] = Field(None, description="New weekday 0=Monday ... 6=Sunday", example=0)
    start_time: Optional[str] = Field(None, description="New start time in HH:MM format", example="18:30")
    end_time: Optional[str] = Field(None, description="New end time in HH:MM format", example="21:00")
    location_zip: Optional[str] = Field(None, description="New ZIP code for this slot", example="94103")


class CauseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Cause identifier", example=2)
    name: str = Field(..., description="Unique cause name", example="animal welfare")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-05T10:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-05T10:00:00Z")


class CausesListResponse(BaseModel):
    causes: List[CauseModel] = Field(..., description="List of causes", example=[])


class OrganizationModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Organization identifier", example=7)
    name: str = Field(..., description="Organization name", example="Chicago Animal Rescue")
    description: Optional[str] = Field(None, description="Short description of the organization", example="Rescuing and rehoming animals in the Chicago area.")
    location_city: Optional[str] = Field(None, description="City", example="Chicago")
    location_state: Optional[str] = Field(None, description="State or region", example="IL")
    location_zip: Optional[str] = Field(None, description="ZIP code", example="60614")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-02T09:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-10T09:00:00Z")


class OrganizationsListResponse(BaseModel):
    organizations: List[OrganizationModel] = Field(..., description="List of organizations", example=[])


class OrganizationContactModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Organization contact identifier", example=3)
    organization_id: int = Field(..., description="Associated organization identifier", example=7)
    name: str = Field(..., description="Contact person name", example="Maria Lopez")
    email: Optional[str] = Field(None, description="Contact email", example="maria.lopez@chicagoanimal.org")
    phone: Optional[str] = Field(None, description="Contact phone number", example="+1-312-555-0187")
    role: Optional[str] = Field(None, description="Role or title at the organization", example="Volunteer Coordinator")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-05T11:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-05T11:00:00Z")


class OrganizationContactsListResponse(BaseModel):
    contacts: List[OrganizationContactModel] = Field(..., description="List of organization contacts", example=[])


class OpportunityModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Opportunity identifier", example=42)
    organization_id: int = Field(..., description="Owning organization identifier", example=7)
    organization_name: Optional[str] = Field(None, description="Name of the owning organization", example="Chicago Animal Rescue")
    title: str = Field(..., description="Opportunity title", example="Weekend Animal Shelter Volunteer")
    description: Optional[str] = Field(None, description="Opportunity description", example="Assist with animal care and adoption events on weekends.")
    is_virtual: bool = Field(..., description="True if opportunity is virtual", example=False)
    location_city: Optional[str] = Field(None, description="City for in-person opportunities", example="Chicago")
    location_state: Optional[str] = Field(None, description="State or region for in-person opportunities", example="IL")
    location_zip: Optional[str] = Field(None, description="ZIP code for in-person opportunities", example="60614")
    cause_id: Optional[int] = Field(None, description="Cause identifier", example=2)
    cause_name: Optional[str] = Field(None, description="Cause name", example="animal welfare")
    min_hours_per_week: Optional[float] = Field(None, description="Minimum hours per week expected", example=2.0)
    max_hours_per_week: Optional[float] = Field(None, description="Maximum hours per week allowed", example=4.0)
    min_commitment_months: Optional[int] = Field(None, description="Minimum suggested commitment in months", example=3)
    start_date: Optional[date] = Field(None, description="Start date of the opportunity", example="2025-02-01")
    end_date: Optional[date] = Field(None, description="End date of the opportunity or null", example="2025-06-30")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-10T12:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-15T09:00:00Z")


class OpportunitiesListResponse(BaseModel):
    opportunities: List[OpportunityModel] = Field(..., description="List of opportunities", example=[])


class OpportunityResponse(BaseModel):
    opportunity: OpportunityModel = Field(..., description="Opportunity object", example={})


class OpportunityShiftModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Shift identifier", example=1001)
    opportunity_id: int = Field(..., description="Associated opportunity identifier", example=42)
    start_datetime: datetime = Field(..., description="Shift start datetime (ISO 8601)", example="2025-02-08T09:00:00Z")
    end_datetime: datetime = Field(..., description="Shift end datetime (ISO 8601)", example="2025-02-08T13:00:00Z")
    max_volunteers: Optional[int] = Field(None, description="Maximum volunteers for this shift or null", example=10)
    notes: Optional[str] = Field(None, description="Additional shift notes", example="Please wear closed-toe shoes.")
    created_at: datetime = Field(..., description="Shift creation timestamp", example="2025-01-15T09:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-16T09:00:00Z")


class OpportunityShiftsListResponse(BaseModel):
    shifts: List[OpportunityShiftModel] = Field(..., description="List of shifts", example=[])


class OpportunitySkillModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    skill_id: int = Field(..., description="Skill identifier", example=3)
    skill_name: str = Field(..., description="Skill name", example="Spanish translation")
    required_level: Optional[str] = Field(None, description="Required level or null if unspecified", example="intermediate")


class OpportunitySkillsListResponse(BaseModel):
    skills: List[OpportunitySkillModel] = Field(..., description="List of opportunity skills", example=[])


class MatchingOpportunityModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Opportunity identifier", example=80)
    title: str = Field(..., description="Opportunity title", example="Remote Grant Research Assistant")
    organization_id: int = Field(..., description="Organization identifier", example=20)
    organization_name: Optional[str] = Field(None, description="Organization name", example="Global Impact Fund")
    is_virtual: bool = Field(..., description="Always true for this endpoint", example=True)
    cause_id: Optional[int] = Field(None, description="Cause identifier", example=4)
    cause_name: Optional[str] = Field(None, description="Cause name", example="poverty alleviation")
    min_commitment_months: Optional[int] = Field(None, description="Minimum commitment months", example=3)
    average_rating: Optional[float] = Field(None, description="Average volunteer rating for this opportunity", example=4.7)
    matching_skill_ids: List[int] = Field(..., description="List of skill IDs where user and opportunity both have the skill", example=[3, 5])


class MatchingOpportunitiesListResponse(BaseModel):
    opportunities: List[MatchingOpportunityModel] = Field(..., description="List of matching opportunities", example=[])


class CreateApplicationRequest(BaseModel):
    message: Optional[str] = Field(None, description="Optional message from the applicant to the organization", example="I have 3 years of tutoring experience and am available on weekday evenings.")
    include_profile_skills: Optional[bool] = Field(None, description="If true, flag that profile skills should be shared", example=True)
    include_profile_availability: Optional[bool] = Field(None, description="If true, flag that profile availability should be shared", example=True)


class ApplicationModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Application identifier", example=200)
    user_id: int = Field(..., description="Applicant user identifier", example=1)
    opportunity_id: int = Field(..., description="Target opportunity identifier", example=55)
    status: str = Field(..., description="Application status", example="pending")
    message: Optional[str] = Field(None, description="Application message content", example="I have 3 years of tutoring experience and am available on weekday evenings.")
    include_profile_skills: bool = Field(..., description="Flag for sharing profile skills", example=True)
    include_profile_availability: bool = Field(..., description="Flag for sharing profile availability", example=True)
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-25T11:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-25T11:00:00Z")


class ApplicationResponse(BaseModel):
    application: ApplicationModel = Field(..., description="Application object", example={})


class MyApplicationModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Application identifier", example=200)
    user_id: int = Field(..., description="User identifier (always 1 here)", example=1)
    opportunity_id: int = Field(..., description="Target opportunity identifier", example=42)
    opportunity_title: Optional[str] = Field(None, description="Title of the opportunity", example="Community Food Pantry Assistant")
    status: str = Field(..., description="Application status", example="waitlisted")
    message: Optional[str] = Field(None, description="Application message", example="I am available every Saturday morning.")
    include_profile_skills: bool = Field(..., description="Whether profile skills are shared", example=True)
    include_profile_availability: bool = Field(..., description="Whether profile availability is shared", example=True)
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-15T12:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-20T09:00:00Z")


class MyApplicationsListResponse(BaseModel):
    applications: List[MyApplicationModel] = Field(..., description="List of my applications", example=[])


class MyApplicationResponse(BaseModel):
    application: MyApplicationModel = Field(..., description="Application object", example={})


class CreateApplicationMessageRequest(BaseModel):
    message: str = Field(..., description="Message body text", example="Please let me know if any additional slots open up; I can also help on Friday mornings.")


class ApplicationMessageModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Application message identifier", example=500)
    application_id: int = Field(..., description="Associated application identifier", example=210)
    sender_type: str = Field(..., description="Sender type ('user' or 'organization')", example="user")
    sender_user_id: Optional[int] = Field(None, description="User sender identifier (1)", example=1)
    organization_contact_id: Optional[int] = Field(None, description="Organization contact sender identifier or null", example=None)
    message: str = Field(..., description="Message content", example="Please let me know if any additional slots open up; I can also help on Friday mornings.")
    created_at: datetime = Field(..., description="Timestamp when the message was created", example="2025-01-20T11:30:00Z")


class ApplicationMessageResponse(BaseModel):
    application_message: ApplicationMessageModel = Field(..., description="Application message", example={})


class ApplicationMessagesListResponse(BaseModel):
    messages: List[ApplicationMessageModel] = Field(..., description="List of messages", example=[])


class RosterModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Roster identifier", example=300)
    user_id: int = Field(..., description="User identifier", example=1)
    opportunity_id: int = Field(..., description="Associated opportunity identifier", example=42)
    opportunity_title: Optional[str] = Field(None, description="Title of the rostered opportunity", example="Senior Center Tech Help Volunteer")
    role: Optional[str] = Field(None, description="Role of the volunteer in this opportunity", example="Volunteer")
    status: str = Field(..., description="Roster status", example="active")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-01-05T12:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-01-10T12:00:00Z")


class RostersListResponse(BaseModel):
    rosters: List[RosterModel] = Field(..., description="List of rosters", example=[])


class RosterResponse(BaseModel):
    roster: RosterModel = Field(..., description="Roster object", example={})


class ShiftAssignmentModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Shift assignment identifier", example=400)
    user_id: int = Field(..., description="User identifier", example=1)
    opportunity_id: int = Field(..., description="Associated opportunity identifier", example=75)
    opportunity_title: Optional[str] = Field(None, description="Title of the opportunity", example="Senior Center Tech Help Volunteer")
    shift_id: int = Field(..., description="Associated shift identifier", example=1001)
    shift_start_datetime: Optional[datetime] = Field(None, description="Shift start datetime (ISO 8601)", example="2025-03-05T10:00:00Z")
    shift_end_datetime: Optional[datetime] = Field(None, description="Shift end datetime (ISO 8601)", example="2025-03-05T12:00:00Z")
    status: str = Field(..., description="Assignment status", example="assigned")
    note: Optional[str] = Field(None, description="Optional note on the assignment", example="Confirmed with site coordinator.")
    created_at: datetime = Field(..., description="Assignment creation timestamp", example="2025-02-20T09:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-02-22T09:00:00Z")


class ShiftAssignmentsListResponse(BaseModel):
    shift_assignments: List[ShiftAssignmentModel] = Field(..., description="List of shift assignments", example=[])


class UpdateShiftAssignmentRequest(BaseModel):
    status: Optional[str] = Field(None, description="New assignment status", example="confirmed")
    note: Optional[str] = Field(None, description="Updated note for this shift assignment", example="I can bring my own laptop if needed.")


class ShiftAssignmentResponse(BaseModel):
    shift_assignment: ShiftAssignmentModel = Field(..., description="Shift assignment object", example={})


class ShiftNoteModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Shift note identifier", example=600)
    shift_assignment_id: int = Field(..., description="Associated shift assignment identifier", example=400)
    user_id: int = Field(..., description="User identifier", example=1)
    note: str = Field(..., description="Note text", example="I can bring my own laptop if needed.")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-02-25T10:05:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-02-25T10:05:00Z")


class CreateShiftNoteRequest(BaseModel):
    note: str = Field(..., description="Note text content", example="I can bring my own laptop if needed.")


class ShiftNoteResponse(BaseModel):
    shift_note: ShiftNoteModel = Field(..., description="Shift note object", example={})


class ShiftNotesListResponse(BaseModel):
    notes: List[ShiftNoteModel] = Field(..., description="List of shift notes", example=[])


class VolunteerHoursCreateRequest(BaseModel):
    opportunity_id: int = Field(..., description="Associated opportunity identifier", example=90)
    shift_id: Optional[int] = Field(None, description="Optional associated shift identifier", example=1100)
    service_date: date = Field(..., description="Date of service (YYYY-MM-DD)", example="2025-10-12", serialization_alias="date")
    start_datetime: Optional[datetime] = Field(None, description="Start datetime of service (ISO 8601)", example="2025-10-12T13:00:00Z")
    end_datetime: Optional[datetime] = Field(None, description="End datetime of service (ISO 8601)", example="2025-10-12T16:30:00Z")
    hours: float = Field(..., description="Total hours served (non-negative)", example=3.5)
    comment: Optional[str] = Field(None, description="Optional comment describing activities performed", example="Helped with meal prep, serving, and cleanup.")


class VolunteerHoursModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Volunteer hours record identifier", example=700)
    user_id: int = Field(..., description="User identifier", example=1)
    opportunity_id: int = Field(..., description="Associated opportunity identifier", example=90)
    shift_id: Optional[int] = Field(None, description="Associated shift identifier or null", example=1100)
    service_date: date = Field(..., description="Date of service", example="2025-10-12", serialization_alias="date")
    start_datetime: Optional[datetime] = Field(None, description="Start datetime or null", example="2025-10-12T13:00:00Z")
    end_datetime: Optional[datetime] = Field(None, description="End datetime or null", example="2025-10-12T16:30:00Z")
    hours: float = Field(..., description="Total hours logged", example=3.5)
    comment: Optional[str] = Field(None, description="Activity comment", example="Helped with meal prep, serving, and cleanup.")
    created_at: datetime = Field(..., description="Creation timestamp", example="2025-10-13T09:00:00Z")
    updated_at: datetime = Field(..., description="Last update timestamp", example="2025-10-13T09:00:00Z")


class VolunteerHoursResponse(BaseModel):
    volunteer_hours: VolunteerHoursModel = Field(..., description="Volunteer hours record", example={})


class VolunteerHoursListItemModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Volunteer hours record identifier", example=700)
    user_id: int = Field(..., description="User identifier", example=1)
    opportunity_id: int = Field(..., description="Associated opportunity identifier", example=90)
    opportunity_title: Optional[str] = Field(None, description="Title of the associated opportunity", example="Homeless Shelter Meal Service")
    service_date: date = Field(..., description="Date of service", example="2025-10-12", serialization_alias="date")
    hours: float = Field(..., description="Hours logged for that date", example=3.5)
    comment: Optional[str] = Field(None, description="Optional comment", example="Helped with meal prep, serving, and cleanup.")


class VolunteerHoursListResponse(BaseModel):
    volunteer_hours: List[VolunteerHoursListItemModel] = Field(..., description="List of volunteer hours", example=[])


class HoursByOpportunityItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    opportunity_id: int = Field(..., description="Opportunity identifier", example=90)
    opportunity_title: Optional[str] = Field(None, description="Opportunity title", example="Homeless Shelter Meal Service")
    cause_id: Optional[int] = Field(None, description="Cause identifier for this opportunity", example=3)
    cause_name: Optional[str] = Field(None, description="Cause name", example="health")
    total_hours: float = Field(..., description="Total hours logged for this opportunity in the year", example=40.0)


class HoursByCauseItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    cause_id: Optional[int] = Field(None, description="Cause identifier", example=1)
    cause_name: Optional[str] = Field(None, description="Cause name", example="education")
    total_hours: float = Field(..., description="Total hours for this cause in the year", example=60.0)


class TopOpportunityItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    opportunity_id: int = Field(..., description="Opportunity identifier", example=90)
    opportunity_title: Optional[str] = Field(None, description="Opportunity title", example="Homeless Shelter Meal Service")
    total_hours: float = Field(..., description="Total hours for this opportunity", example=40.0)


class HoursSummaryResponse(BaseModel):
    year: int = Field(..., description="Year used for the aggregation", example=2025)
    total_hours: float = Field(..., description="Total volunteer hours in this year", example=120.5)
    hours_by_opportunity: List[HoursByOpportunityItem] = Field(..., description="Hours grouped by opportunity", example=[])
    hours_by_cause: List[HoursByCauseItem] = Field(..., description="Hours grouped by cause", example=[])
    top_opportunities: List[TopOpportunityItem] = Field(..., description="Top opportunities by hours", example=[])


class OpportunityRatingsSummaryResponse(BaseModel):
    opportunity_id: int = Field(..., description="Opportunity identifier", example=80)
    average_rating: Optional[float] = Field(None, description="Average rating value between 1 and 5, or null if no ratings", example=4.5)
    ratings_count: int = Field(..., description="Number of ratings submitted for this opportunity", example=12)


class FavoriteModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Favorite record identifier", example=900)
    user_id: int = Field(..., description="User identifier", example=1)
    opportunity_id: int = Field(..., description="Favorited opportunity identifier", example=80)
    opportunity_title: Optional[str] = Field(None, description="Title of the favorited opportunity", example="Remote Grant Research Assistant")
    note: Optional[str] = Field(None, description="User note about this favorite", example="Consider for long-term remote role in the next quarter.")
    created_at: datetime = Field(..., description="When this favorite was created", example="2025-02-01T09:00:00Z")
    updated_at: datetime = Field(..., description="When this favorite was last updated", example="2025-02-01T09:00:00Z")


class FavoritesListResponse(BaseModel):
    favorites: List[FavoriteModel] = Field(..., description="List of favorites", example=[])


class CreateFavoriteRequest(BaseModel):
    opportunity_id: int = Field(..., description="Opportunity identifier to mark as favorite", example=80)
    note: Optional[str] = Field(None, description="Optional note about this favorite", example="Consider for long-term remote role in the next quarter.")


class FavoriteResponse(BaseModel):
    favorite: FavoriteModel = Field(..., description="Favorite record", example={})


class UpdateFavoriteRequest(BaseModel):
    note: str = Field(..., description="New note content", example="High priority for next quarter applications.")


class ShiftFilteredOpportunityModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Opportunity identifier", example=42)
    title: str = Field(..., description="Opportunity title", example="Weekend Animal Shelter Volunteer")
    organization_id: int = Field(..., description="Organization identifier", example=7)
    organization_name: Optional[str] = Field(None, description="Organization name", example="Chicago Animal Rescue")
    start_date: Optional[date] = Field(None, description="Opportunity start date", example="2025-02-01")
    next_shift_start_datetime: Optional[datetime] = Field(None, description="Earliest matching shift start datetime", example="2025-02-01T09:00:00Z")


class ShiftFilteredOpportunitiesResponse(BaseModel):
    opportunities: List[ShiftFilteredOpportunityModel] = Field(..., description="Opportunities with matching shifts", example=[])


class LastCompletedShiftModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., description="Shift assignment identifier", example=450)
    shift_id: int = Field(..., description="Associated shift identifier", example=1200)
    shift_start_datetime: Optional[datetime] = Field(None, description="Shift start datetime", example="2025-03-20T18:00:00Z")
    shift_end_datetime: Optional[datetime] = Field(None, description="Shift end datetime", example="2025-03-20T20:00:00Z")
    status: str = Field(..., description="Assignment status (completed)", example="completed")


class LastCompletedShiftResponse(BaseModel):
    shift_assignment: Optional[LastCompletedShiftModel] = Field(None, description="Last completed shift assignment", example={})


@app.get(
    "/api/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Retrieve the profile of the authenticated user with basic info.",
    tags=["users"],
    operation_id="get_current_user_profile",
)
async def get_current_user_profile() -> UserResponse:
    session = SessionLocal()
    user = session.query(User).filter(User.id == 1).first()
    if user is None:
        user = User(
            id=1,
            username="user1",
            email="user1@example.org",
            full_name=None,
            location_city=None,
            location_state=None,
            location_zip=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(user)
        session.commit()
    response = UserResponse(user=user)
    session.close()
    return response


@app.patch(
    "/api/me",
    response_model=UserResponse,
    summary="Update current user profile",
    description="Partially update profile fields for the authenticated user.",
    tags=["users"],
    operation_id="update_current_user_profile",
)
async def update_current_user_profile(body: UpdateUserRequest = Body(...)) -> UserResponse:
    session = SessionLocal()
    user = session.query(User).filter(User.id == 1).first()
    if user is None:
        user = User(
            id=1,
            username="user1",
            email="user1@example.org",
            full_name=None,
            location_city=None,
            location_state=None,
            location_zip=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(user)
        session.commit()
    if body.full_name is not None:
        user.full_name = body.full_name
    if body.location_city is not None:
        user.location_city = body.location_city
    if body.location_state is not None:
        user.location_state = body.location_state
    if body.location_zip is not None:
        user.location_zip = body.location_zip
    user.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(user)
    response = UserResponse(user=user)
    session.close()
    return response


@app.get(
    "/api/skills",
    response_model=SkillsListResponse,
    summary="List all skills",
    description="Retrieve all skills with optional search and pagination.",
    tags=["skills"],
    operation_id="list_skills",
)
async def list_skills(
    search: Optional[str] = Query(None, description="Optional case-insensitive partial name filter", example="tutor"),
    limit: Optional[int] = Query(None, description="Maximum number of skills to return", example=50),
    offset: Optional[int] = Query(None, description="Offset for pagination", example=0),
) -> SkillsListResponse:
    session = SessionLocal()
    query = session.query(Skill)
    if search is not None:
        pattern = f"%{search}%"
        query = query.filter(Skill.name.ilike(pattern))
    if offset is not None:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)
    skills = query.all()
    response = SkillsListResponse(skills=skills)
    session.close()
    return response


@app.post(
    "/api/skills",
    response_model=SkillResponse,
    summary="Create a new skill",
    description="Create a new skill name if it does not already exist.",
    tags=["skills"],
    operation_id="create_skill",
)
async def create_skill(body: CreateSkillRequest = Body(...)) -> SkillResponse:
    session = SessionLocal()
    existing = session.query(Skill).filter(Skill.name == body.name).first()
    if existing is not None:
        response = SkillResponse(skill=existing)
        session.close()
        return response
    now = datetime.utcnow()
    skill = Skill(name=body.name, created_at=now, updated_at=now)
    session.add(skill)
    session.commit()
    session.refresh(skill)
    response = SkillResponse(skill=skill)
    session.close()
    return response


@app.get(
    "/api/me/skills",
    response_model=UserSkillsListResponse,
    summary="List my skills",
    description="Retrieve all skills associated with the authenticated user.",
    tags=["user_skills"],
    operation_id="list_my_skills",
)
async def list_my_skills() -> UserSkillsListResponse:
    session = SessionLocal()
    records = (
        session.query(UserSkill, Skill)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .filter(UserSkill.user_id == 1)
        .all()
    )
    items: List[UserSkillModel] = []
    for us, sk in records:
        item = UserSkillModel(
            skill_id=us.skill_id,
            skill_name=sk.name,
            proficiency_level=us.proficiency_level,
            created_at=us.created_at,
            updated_at=us.updated_at,
        )
        items.append(item)
    response = UserSkillsListResponse(skills=items)
    session.close()
    return response


@app.post(
    "/api/me/skills",
    response_model=UserSkillResponse,
    summary="Add a skill to my profile",
    description="Associate an existing skill with the authenticated user.",
    tags=["user_skills"],
    operation_id="add_skill_to_my_profile",
)
async def add_skill_to_my_profile(body: AddUserSkillRequest = Body(...)) -> UserSkillResponse:
    session = SessionLocal()
    user_skill = session.query(UserSkill).filter(UserSkill.user_id == 1, UserSkill.skill_id == body.skill_id).first()
    now = datetime.utcnow()
    if user_skill is None:
        user_skill = UserSkill(
            user_id=1,
            skill_id=body.skill_id,
            proficiency_level=body.proficiency_level,
            created_at=now,
            updated_at=now,
        )
        session.add(user_skill)
    else:
        if body.proficiency_level is not None:
            user_skill.proficiency_level = body.proficiency_level
        user_skill.updated_at = now
    session.commit()
    session.refresh(user_skill)
    record = UserSkillRecordModel(
        user_id=user_skill.user_id,
        skill_id=user_skill.skill_id,
        proficiency_level=user_skill.proficiency_level,
        created_at=user_skill.created_at,
        updated_at=user_skill.updated_at,
    )
    response = UserSkillResponse(user_skill=record)
    session.close()
    return response


@app.patch(
    "/api/me/skills/{skill_id}",
    response_model=UserSkillResponse,
    summary="Update my proficiency for a skill",
    description="Update the proficiency_level for a specific skill linked to the user.",
    tags=["user_skills"],
    operation_id="update_my_skill_proficiency",
)
async def update_my_skill_proficiency(
    skill_id: int = Path(..., description="Skill identifier to update", example=3),
    body: UpdateUserSkillRequest = Body(...),
) -> UserSkillResponse:
    session = SessionLocal()
    user_skill = session.query(UserSkill).filter(UserSkill.user_id == 1, UserSkill.skill_id == skill_id).first()
    now = datetime.utcnow()
    if user_skill is None:
        user_skill = UserSkill(
            user_id=1,
            skill_id=skill_id,
            proficiency_level=body.proficiency_level,
            created_at=now,
            updated_at=now,
        )
        session.add(user_skill)
    else:
        user_skill.proficiency_level = body.proficiency_level
        user_skill.updated_at = now
    session.commit()
    session.refresh(user_skill)
    record = UserSkillRecordModel(
        user_id=user_skill.user_id,
        skill_id=user_skill.skill_id,
        proficiency_level=user_skill.proficiency_level,
        created_at=user_skill.created_at,
        updated_at=user_skill.updated_at,
    )
    response = UserSkillResponse(user_skill=record)
    session.close()
    return response


@app.delete(
    "/api/me/skills/{skill_id}",
    response_model=DeleteSuccessResponse,
    summary="Remove a skill from my profile",
    description="Delete the association between the user and the given skill_id.",
    tags=["user_skills"],
    operation_id="remove_skill_from_my_profile",
)
async def remove_skill_from_my_profile(
    skill_id: int = Path(..., description="Skill identifier to remove from user", example=5)
) -> DeleteSuccessResponse:
    session = SessionLocal()
    user_skill = session.query(UserSkill).filter(UserSkill.user_id == 1, UserSkill.skill_id == skill_id).first()
    success = False
    if user_skill is not None:
        session.delete(user_skill)
        session.commit()
        success = True
    session.close()
    return DeleteSuccessResponse(success=success)


@app.get(
    "/api/me/availability",
    response_model=UserAvailabilityListResponse,
    summary="List my availability slots",
    description="Retrieve all availability windows configured for the user.",
    tags=["user_availability"],
    operation_id="list_my_availability",
)
async def list_my_availability() -> UserAvailabilityListResponse:
    session = SessionLocal()
    records = session.query(UserAvailability).filter(UserAvailability.user_id == 1).all()
    response = UserAvailabilityListResponse(availability=records)
    session.close()
    return response


@app.post(
    "/api/me/availability",
    response_model=UserAvailabilityResponse,
    summary="Create an availability slot",
    description="Add a new weekly availability window for the user.",
    tags=["user_availability"],
    operation_id="create_my_availability",
)
async def create_my_availability(body: CreateAvailabilityRequest = Body(...)) -> UserAvailabilityResponse:
    session = SessionLocal()
    now = datetime.utcnow()
    availability = UserAvailability(
        user_id=1,
        weekday=body.weekday,
        start_time=body.start_time,
        end_time=body.end_time,
        location_zip=body.location_zip,
        created_at=now,
        updated_at=now,
    )
    session.add(availability)
    session.commit()
    session.refresh(availability)
    response = UserAvailabilityResponse(availability=availability)
    session.close()
    return response


@app.patch(
    "/api/me/availability/{availability_id}",
    response_model=UserAvailabilityResponse,
    summary="Update an availability slot",
    description="Partially update an existing availability window for the user.",
    tags=["user_availability"],
    operation_id="update_my_availability",
)
async def update_my_availability(
    availability_id: int = Path(..., description="Availability record identifier to update", example=10),
    body: UpdateAvailabilityRequest = Body(...),
) -> UserAvailabilityResponse:
    session = SessionLocal()
    availability = (
        session.query(UserAvailability)
        .filter(UserAvailability.id == availability_id, UserAvailability.user_id == 1)
        .first()
    )
    now = datetime.utcnow()
    if availability is None:
        availability = UserAvailability(
            id=availability_id,
            user_id=1,
            weekday=body.weekday if body.weekday is not None else 0,
            start_time=body.start_time if body.start_time is not None else "00:00",
            end_time=body.end_time if body.end_time is not None else "00:00",
            location_zip=body.location_zip,
            created_at=now,
            updated_at=now,
        )
        session.add(availability)
    else:
        if body.weekday is not None:
            availability.weekday = body.weekday
        if body.start_time is not None:
            availability.start_time = body.start_time
        if body.end_time is not None:
            availability.end_time = body.end_time
        if body.location_zip is not None:
            availability.location_zip = body.location_zip
        availability.updated_at = now
    session.commit()
    session.refresh(availability)
    response = UserAvailabilityResponse(availability=availability)
    session.close()
    return response


@app.delete(
    "/api/me/availability/{availability_id}",
    response_model=DeleteSuccessResponse,
    summary="Delete an availability slot",
    description="Remove an availability record belonging to the user.",
    tags=["user_availability"],
    operation_id="delete_my_availability",
)
async def delete_my_availability(
    availability_id: int = Path(..., description="Availability record identifier to delete", example=11)
) -> DeleteSuccessResponse:
    session = SessionLocal()
    availability = (
        session.query(UserAvailability)
        .filter(UserAvailability.id == availability_id, UserAvailability.user_id == 1)
        .first()
    )
    success = False
    if availability is not None:
        session.delete(availability)
        session.commit()
        success = True
    session.close()
    return DeleteSuccessResponse(success=success)


@app.get(
    "/api/causes",
    response_model=CausesListResponse,
    summary="List all causes",
    description="Retrieve a list of all cause areas available.",
    tags=["causes"],
    operation_id="list_causes",
)
async def list_causes() -> CausesListResponse:
    session = SessionLocal()
    causes = session.query(Cause).all()
    response = CausesListResponse(causes=causes)
    session.close()
    return response


@app.get(
    "/api/organizations",
    response_model=OrganizationsListResponse,
    summary="List organizations with filters",
    description="Retrieve organizations with optional search and location filters.",
    tags=["organizations"],
    operation_id="list_organizations",
)
async def list_organizations(
    search: Optional[str] = Query(None, description="Case-insensitive partial name search", example="Humane Society"),
    location_city: Optional[str] = Query(None, description="Filter by city", example="Chicago"),
    location_state: Optional[str] = Query(None, description="Filter by state or region", example="IL"),
    limit: Optional[int] = Query(None, description="Maximum number of organizations", example=20),
    offset: Optional[int] = Query(None, description="Offset for pagination", example=0),
) -> OrganizationsListResponse:
    session = SessionLocal()
    query = session.query(Organization)
    if search is not None:
        pattern = f"%{search}%"
        query = query.filter(Organization.name.ilike(pattern))
    if location_city is not None:
        query = query.filter(Organization.location_city == location_city)
    if location_state is not None:
        query = query.filter(Organization.location_state == location_state)
    if offset is not None:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)
    organizations = query.all()
    response = OrganizationsListResponse(organizations=organizations)
    session.close()
    return response


@app.get(
    "/api/organizations/{organization_id}/contacts",
    response_model=OrganizationContactsListResponse,
    summary="List contacts for an organization",
    description="Retrieve all contacts for a given organization.",
    tags=["organization_contacts"],
    operation_id="list_organization_contacts",
)
async def list_organization_contacts(
    organization_id: int = Path(..., description="Organization identifier", example=7)
) -> OrganizationContactsListResponse:
    session = SessionLocal()
    contacts = (
        session.query(OrganizationContact)
        .filter(OrganizationContact.organization_id == organization_id)
        .all()
    )
    response = OrganizationContactsListResponse(contacts=contacts)
    session.close()
    return response


@app.get(
    "/api/opportunities",
    response_model=OpportunitiesListResponse,
    summary="Search opportunities with filters",
    description="Retrieve opportunities using multiple filters and optional ordering.",
    tags=["opportunities"],
    operation_id="search_opportunities",
)
async def search_opportunities(
    title_search: Optional[str] = Query(None, description="Case-insensitive partial search on opportunity title", example="ESL Tutor"),
    location_city: Optional[str] = Query(None, description="Filter by city", example="Chicago"),
    location_state: Optional[str] = Query(None, description="Filter by state or region", example="IL"),
    cause_name: Optional[str] = Query(None, description="Filter by cause name", example="animal welfare"),
    is_virtual: Optional[bool] = Query(None, description="Filter by virtual opportunities", example=False),
    max_hours_per_week_lte: Optional[float] = Query(None, description="Filter max_hours_per_week <= value", example=4.0),
    min_commitment_months_gte: Optional[int] = Query(None, description="Filter min_commitment_months >= value", example=2),
    order_by_start_date: Optional[bool] = Query(None, description="If true, order by ascending start_date", example=True),
    limit: Optional[int] = Query(None, description="Maximum number of opportunities to return", example=10),
    offset: Optional[int] = Query(None, description="Offset for pagination", example=0),
) -> OpportunitiesListResponse:
    session = SessionLocal()
    query = session.query(Opportunity, Organization, Cause).join(Organization, Organization.id == Opportunity.organization_id).outerjoin(Cause, Cause.id == Opportunity.cause_id)
    if title_search is not None:
        pattern = f"%{title_search}%"
        query = query.filter(Opportunity.title.ilike(pattern))
    if location_city is not None:
        query = query.filter(Opportunity.location_city == location_city)
    if location_state is not None:
        query = query.filter(Opportunity.location_state == location_state)
    if cause_name is not None:
        pattern_c = f"%{cause_name}%"
        query = query.filter(Cause.name.ilike(pattern_c))
    if is_virtual is not None:
        query = query.filter(Opportunity.is_virtual == (1 if is_virtual else 0))
    if max_hours_per_week_lte is not None:
        query = query.filter(Opportunity.max_hours_per_week <= max_hours_per_week_lte)
    if min_commitment_months_gte is not None:
        query = query.filter(Opportunity.min_commitment_months >= min_commitment_months_gte)
    if order_by_start_date:
        query = query.order_by(Opportunity.start_date.asc())
    if offset is not None:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)
    rows = query.all()
    items: List[OpportunityModel] = []
    for opp, org, cause in rows:
        item = OpportunityModel(
            id=opp.id,
            organization_id=opp.organization_id,
            organization_name=org.name if org is not None else None,
            title=opp.title,
            description=opp.description,
            is_virtual=bool(opp.is_virtual),
            location_city=opp.location_city,
            location_state=opp.location_state,
            location_zip=opp.location_zip,
            cause_id=opp.cause_id,
            cause_name=cause.name if cause is not None else None,
            min_hours_per_week=opp.min_hours_per_week,
            max_hours_per_week=opp.max_hours_per_week,
            min_commitment_months=opp.min_commitment_months,
            start_date=opp.start_date,
            end_date=opp.end_date,
            created_at=opp.created_at,
            updated_at=opp.updated_at,
        )
        items.append(item)
    response = OpportunitiesListResponse(opportunities=items)
    session.close()
    return response


@app.get(
    "/api/opportunities/{opportunity_id}",
    response_model=OpportunityResponse,
    summary="Get a single opportunity by ID",
    description="Retrieve detailed information for a single opportunity record.",
    tags=["opportunities"],
    operation_id="get_opportunity_by_id",
)
async def get_opportunity_by_id(
    opportunity_id: int = Path(..., description="Opportunity identifier", example=42)
) -> OpportunityResponse:
    session = SessionLocal()
    row = (
        session.query(Opportunity, Organization, Cause)
        .join(Organization, Organization.id == Opportunity.organization_id)
        .outerjoin(Cause, Cause.id == Opportunity.cause_id)
        .filter(Opportunity.id == opportunity_id)
        .first()
    )
    if row is None:
        opp = Opportunity(
            id=opportunity_id,
            organization_id=1,
            title="",
            description=None,
            is_virtual=0,
            location_city=None,
            location_state=None,
            location_zip=None,
            cause_id=None,
            min_hours_per_week=None,
            max_hours_per_week=None,
            min_commitment_months=None,
            start_date=None,
            end_date=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        org = session.query(Organization).filter(Organization.id == opp.organization_id).first()
        cause = None
    else:
        opp, org, cause = row
    item = OpportunityModel(
        id=opp.id,
        organization_id=opp.organization_id,
        organization_name=org.name if org is not None else None,
        title=opp.title,
        description=opp.description,
        is_virtual=bool(opp.is_virtual),
        location_city=opp.location_city,
        location_state=opp.location_state,
        location_zip=opp.location_zip,
        cause_id=opp.cause_id,
        cause_name=cause.name if cause is not None else None,
        min_hours_per_week=opp.min_hours_per_week,
        max_hours_per_week=opp.max_hours_per_week,
        min_commitment_months=opp.min_commitment_months,
        start_date=opp.start_date,
        end_date=opp.end_date,
        created_at=opp.created_at,
        updated_at=opp.updated_at,
    )
    response = OpportunityResponse(opportunity=item)
    session.close()
    return response


@app.get(
    "/api/opportunities/by-title",
    response_model=OpportunityResponse,
    summary="Get single opportunity by exact title",
    description="Retrieve one opportunity matching an exact title.",
    tags=["opportunities"],
    operation_id="get_opportunity_by_title",
)
async def get_opportunity_by_title(
    title: str = Query(..., description="Exact opportunity title to look up", example="Virtual ESL Tutor for Adult Learners")
) -> OpportunityResponse:
    session = SessionLocal()
    row = (
        session.query(Opportunity, Organization, Cause)
        .join(Organization, Organization.id == Opportunity.organization_id)
        .outerjoin(Cause, Cause.id == Opportunity.cause_id)
        .filter(Opportunity.title == title)
        .first()
    )
    if row is None:
        opp = Opportunity(
            id=0,
            organization_id=1,
            title=title,
            description=None,
            is_virtual=0,
            location_city=None,
            location_state=None,
            location_zip=None,
            cause_id=None,
            min_hours_per_week=None,
            max_hours_per_week=None,
            min_commitment_months=None,
            start_date=None,
            end_date=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        org = session.query(Organization).filter(Organization.id == opp.organization_id).first()
        cause = None
    else:
        opp, org, cause = row
    item = OpportunityModel(
        id=opp.id,
        organization_id=opp.organization_id,
        organization_name=org.name if org is not None else None,
        title=opp.title,
        description=opp.description,
        is_virtual=bool(opp.is_virtual),
        location_city=opp.location_city,
        location_state=opp.location_state,
        location_zip=opp.location_zip,
        cause_id=opp.cause_id,
        cause_name=cause.name if cause is not None else None,
        min_hours_per_week=opp.min_hours_per_week,
        max_hours_per_week=opp.max_hours_per_week,
        min_commitment_months=opp.min_commitment_months,
        start_date=opp.start_date,
        end_date=opp.end_date,
        created_at=opp.created_at,
        updated_at=opp.updated_at,
    )
    response = OpportunityResponse(opportunity=item)
    session.close()
    return response


@app.get(
    "/api/opportunities/{opportunity_id}/shifts",
    response_model=OpportunityShiftsListResponse,
    summary="List shifts for an opportunity",
    description="Retrieve all shifts associated with a specific opportunity.",
    tags=["opportunity_shifts"],
    operation_id="list_opportunity_shifts",
)
async def list_opportunity_shifts(
    opportunity_id: int = Path(..., description="Opportunity identifier to list shifts for", example=42),
    start_from: Optional[datetime] = Query(None, description="Include shifts with start_datetime >= this timestamp", example="2025-02-01T00:00:00Z"),
    start_to: Optional[datetime] = Query(None, description="Include shifts with start_datetime <= this timestamp", example="2025-02-28T23:59:59Z"),
) -> OpportunityShiftsListResponse:
    session = SessionLocal()
    query = session.query(OpportunityShift).filter(OpportunityShift.opportunity_id == opportunity_id)
    if start_from is not None:
        query = query.filter(OpportunityShift.start_datetime >= start_from)
    if start_to is not None:
        query = query.filter(OpportunityShift.start_datetime <= start_to)
    shifts = query.all()
    response = OpportunityShiftsListResponse(shifts=shifts)
    session.close()
    return response


@app.get(
    "/api/opportunities/{opportunity_id}/skills",
    response_model=OpportunitySkillsListResponse,
    summary="List skills required for an opportunity",
    description="Retrieve the skills linked to an opportunity with required levels.",
    tags=["opportunity_skills"],
    operation_id="list_opportunity_skills",
)
async def list_opportunity_skills(
    opportunity_id: int = Path(..., description="Opportunity identifier", example=55)
) -> OpportunitySkillsListResponse:
    session = SessionLocal()
    rows = (
        session.query(OpportunitySkill, Skill)
        .join(Skill, Skill.id == OpportunitySkill.skill_id)
        .filter(OpportunitySkill.opportunity_id == opportunity_id)
        .all()
    )
    items: List[OpportunitySkillModel] = []
    for osk, sk in rows:
        item = OpportunitySkillModel(skill_id=osk.skill_id, skill_name=sk.name, required_level=osk.required_level)
        items.append(item)
    response = OpportunitySkillsListResponse(skills=items)
    session.close()
    return response


@app.get(
    "/api/opportunities/virtual/match-my-skills",
    response_model=MatchingOpportunitiesListResponse,
    summary="Find virtual opportunities matching my skills",
    description="Return virtual opportunities matching my skills and minimum commitment.",
    tags=["opportunities", "matching"],
    operation_id="find_virtual_opportunities_matching_my_skills",
)
async def find_virtual_opportunities_matching_my_skills(
    min_matching_skills: Optional[int] = Query(2, description="Minimum number of overlapping skills with my profile", example=2),
    min_commitment_months: Optional[int] = Query(None, description="Minimum value of min_commitment_months to include", example=2),
    limit: Optional[int] = Query(None, description="Maximum number of opportunities to return", example=10),
) -> MatchingOpportunitiesListResponse:
    session = SessionLocal()
    my_skill_rows = session.query(UserSkill.skill_id).filter(UserSkill.user_id == 1).all()
    my_skill_ids = {row[0] for row in my_skill_rows}
    opp_skill_rows = session.query(OpportunitySkill.opportunity_id, OpportunitySkill.skill_id).all()
    opp_to_skills: Dict[int, List[int]] = {}
    for oid, sid in opp_skill_rows:
        if oid not in opp_to_skills:
            opp_to_skills[oid] = []
        opp_to_skills[oid].append(sid)
    rating_rows = (
        session.query(OpportunityRating.opportunity_id, OpportunityRating.rating)
        .all()
    )
    rating_sum: Dict[int, float] = {}
    rating_count: Dict[int, int] = {}
    for oid, rating in rating_rows:
        rating_sum[oid] = rating_sum.get(oid, 0.0) + float(rating)
        rating_count[oid] = rating_count.get(oid, 0) + 1
    query = session.query(Opportunity, Organization, Cause).join(Organization, Organization.id == Opportunity.organization_id).outerjoin(Cause, Cause.id == Opportunity.cause_id)
    query = query.filter(Opportunity.is_virtual == 1)
    if min_commitment_months is not None:
        query = query.filter(Opportunity.min_commitment_months >= min_commitment_months)
    rows = query.all()
    items: List[MatchingOpportunityModel] = []
    for opp, org, cause in rows:
        skills_for_opp = opp_to_skills.get(opp.id, [])
        matching = [sid for sid in skills_for_opp if sid in my_skill_ids]
        if len(matching) >= (min_matching_skills if min_matching_skills is not None else 0):
            count = rating_count.get(opp.id, 0)
            avg = None
            if count > 0:
                avg = rating_sum[opp.id] / float(count)
            item = MatchingOpportunityModel(
                id=opp.id,
                title=opp.title,
                organization_id=opp.organization_id,
                organization_name=org.name if org is not None else None,
                is_virtual=True,
                cause_id=opp.cause_id,
                cause_name=cause.name if cause is not None else None,
                min_commitment_months=opp.min_commitment_months,
                average_rating=avg,
                matching_skill_ids=matching,
            )
            items.append(item)
    items.sort(key=lambda x: (x.average_rating if x.average_rating is not None else 0.0), reverse=True)
    if limit is not None:
        items = items[:limit]
    response = MatchingOpportunitiesListResponse(opportunities=items)
    session.close()
    return response


@app.post(
    "/api/opportunities/{opportunity_id}/applications",
    response_model=ApplicationResponse,
    summary="Submit an application to an opportunity",
    description="Create a new application by the authenticated user for the opportunity.",
    tags=["applications"],
    operation_id="create_application_for_opportunity",
)
async def create_application_for_opportunity(
    opportunity_id: int = Path(..., description="Target opportunity identifier", example=55),
    body: CreateApplicationRequest = Body(...),
) -> ApplicationResponse:
    session = SessionLocal()
    include_skills = bool(body.include_profile_skills) if body.include_profile_skills is not None else False
    include_availability = bool(body.include_profile_availability) if body.include_profile_availability is not None else False
    now = datetime.utcnow()
    application = Application(
        user_id=1,
        opportunity_id=opportunity_id,
        status="pending",
        message=body.message,
        include_profile_skills=1 if include_skills else 0,
        include_profile_availability=1 if include_availability else 0,
        created_at=now,
        updated_at=now,
    )
    session.add(application)
    session.commit()
    session.refresh(application)
    model = ApplicationModel(
        id=application.id,
        user_id=application.user_id,
        opportunity_id=application.opportunity_id,
        status=application.status,
        message=application.message,
        include_profile_skills=bool(application.include_profile_skills),
        include_profile_availability=bool(application.include_profile_availability),
        created_at=application.created_at,
        updated_at=application.updated_at,
    )
    response = ApplicationResponse(application=model)
    session.close()
    return response


@app.get(
    "/api/me/applications",
    response_model=MyApplicationsListResponse,
    summary="List my applications with filters",
    description="Retrieve applications created by the authenticated user.",
    tags=["applications"],
    operation_id="list_my_applications",
)
async def list_my_applications(
    status: Optional[str] = Query(None, description="Filter by application status", example="pending"),
    opportunity_id: Optional[int] = Query(None, description="Filter by specific opportunity identifier", example=42),
) -> MyApplicationsListResponse:
    session = SessionLocal()
    query = session.query(Application, Opportunity).join(Opportunity, Opportunity.id == Application.opportunity_id).filter(Application.user_id == 1)
    if status is not None:
        query = query.filter(Application.status == status)
    if opportunity_id is not None:
        query = query.filter(Application.opportunity_id == opportunity_id)
    rows = query.all()
    items: List[MyApplicationModel] = []
    for app_row, opp in rows:
        item = MyApplicationModel(
            id=app_row.id,
            user_id=app_row.user_id,
            opportunity_id=app_row.opportunity_id,
            opportunity_title=opp.title if opp is not None else None,
            status=app_row.status,
            message=app_row.message,
            include_profile_skills=bool(app_row.include_profile_skills),
            include_profile_availability=bool(app_row.include_profile_availability),
            created_at=app_row.created_at,
            updated_at=app_row.updated_at,
        )
        items.append(item)
    response = MyApplicationsListResponse(applications=items)
    session.close()
    return response


@app.get(
    "/api/me/applications/{application_id}",
    response_model=MyApplicationResponse,
    summary="Get one of my applications by ID",
    description="Retrieve a single application belonging to the authenticated user.",
    tags=["applications"],
    operation_id="get_my_application_by_id",
)
async def get_my_application_by_id(
    application_id: int = Path(..., description="Application identifier", example=210)
) -> MyApplicationResponse:
    session = SessionLocal()
    row = (
        session.query(Application, Opportunity)
        .join(Opportunity, Opportunity.id == Application.opportunity_id)
        .filter(Application.id == application_id, Application.user_id == 1)
        .first()
    )
    if row is None:
        application = Application(
            id=application_id,
            user_id=1,
            opportunity_id=0,
            status="pending",
            message=None,
            include_profile_skills=0,
            include_profile_availability=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        opp = None
    else:
        application, opp = row
    item = MyApplicationModel(
        id=application.id,
        user_id=application.user_id,
        opportunity_id=application.opportunity_id,
        opportunity_title=opp.title if opp is not None else None,
        status=application.status,
        message=application.message,
        include_profile_skills=bool(application.include_profile_skills),
        include_profile_availability=bool(application.include_profile_availability),
        created_at=application.created_at,
        updated_at=application.updated_at,
    )
    response = MyApplicationResponse(application=item)
    session.close()
    return response


@app.get(
    "/api/me/applications/by-opportunity-title",
    response_model=MyApplicationResponse,
    summary="Get my application by opportunity title",
    description="Find the user's application for an opportunity with a specific title.",
    tags=["applications"],
    operation_id="get_my_application_by_opportunity_title",
)
async def get_my_application_by_opportunity_title(
    opportunity_title: str = Query(..., description="Exact opportunity title to find the application for", example="Community Food Pantry Assistant")
) -> MyApplicationResponse:
    session = SessionLocal()
    row = (
        session.query(Application, Opportunity)
        .join(Opportunity, Opportunity.id == Application.opportunity_id)
        .filter(Application.user_id == 1, Opportunity.title == opportunity_title)
        .first()
    )
    if row is None:
        application = Application(
            id=0,
            user_id=1,
            opportunity_id=0,
            status="pending",
            message=None,
            include_profile_skills=0,
            include_profile_availability=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        opp = None
    else:
        application, opp = row
    item = MyApplicationModel(
        id=application.id,
        user_id=application.user_id,
        opportunity_id=application.opportunity_id,
        opportunity_title=opp.title if opp is not None else None,
        status=application.status,
        message=application.message,
        include_profile_skills=bool(application.include_profile_skills),
        include_profile_availability=bool(application.include_profile_availability),
        created_at=application.created_at,
        updated_at=application.updated_at,
    )
    response = MyApplicationResponse(application=item)
    session.close()
    return response


@app.post(
    "/api/me/applications/{application_id}/messages",
    response_model=ApplicationMessageResponse,
    summary="Send a message on an application thread",
    description="Create a new message from the user within an application conversation.",
    tags=["application_messages"],
    operation_id="send_application_message",
)
async def send_application_message(
    application_id: int = Path(..., description="Application identifier to attach the message to", example=210),
    body: CreateApplicationMessageRequest = Body(...),
) -> ApplicationMessageResponse:
    session = SessionLocal()
    now = datetime.utcnow()
    message = ApplicationMessage(
        application_id=application_id,
        sender_type="user",
        sender_user_id=1,
        organization_contact_id=None,
        message=body.message,
        created_at=now,
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    model = ApplicationMessageModel(
        id=message.id,
        application_id=message.application_id,
        sender_type=message.sender_type,
        sender_user_id=message.sender_user_id,
        organization_contact_id=message.organization_contact_id,
        message=message.message,
        created_at=message.created_at,
    )
    response = ApplicationMessageResponse(application_message=model)
    session.close()
    return response


@app.get(
    "/api/me/applications/{application_id}/messages",
    response_model=ApplicationMessagesListResponse,
    summary="List messages for one of my applications",
    description="Retrieve the message thread for an application belonging to the user.",
    tags=["application_messages"],
    operation_id="list_application_messages_for_me",
)
async def list_application_messages_for_me(
    application_id: int = Path(..., description="Application identifier", example=210)
) -> ApplicationMessagesListResponse:
    session = SessionLocal()
    application = session.query(Application).filter(Application.id == application_id, Application.user_id == 1).first()
    if application is None:
        messages: List[ApplicationMessageModel] = []
        response = ApplicationMessagesListResponse(messages=messages)
        session.close()
        return response
    rows = (
        session.query(ApplicationMessage)
        .filter(ApplicationMessage.application_id == application_id)
        .order_by(ApplicationMessage.created_at.asc())
        .all()
    )
    items: List[ApplicationMessageModel] = []
    for msg in rows:
        item = ApplicationMessageModel(
            id=msg.id,
            application_id=msg.application_id,
            sender_type=msg.sender_type,
            sender_user_id=msg.sender_user_id,
            organization_contact_id=msg.organization_contact_id,
            message=msg.message,
            created_at=msg.created_at,
        )
        items.append(item)
    response = ApplicationMessagesListResponse(messages=items)
    session.close()
    return response


@app.get(
    "/api/me/rosters",
    response_model=RostersListResponse,
    summary="List opportunities where I am on the roster",
    description="Retrieve all roster entries for the authenticated user.",
    tags=["rosters"],
    operation_id="list_my_rosters",
)
async def list_my_rosters(
    status: Optional[str] = Query(None, description="Filter rosters by status (active/inactive)", example="active")
) -> RostersListResponse:
    session = SessionLocal()
    query = session.query(Roster, Opportunity).join(Opportunity, Opportunity.id == Roster.opportunity_id).filter(Roster.user_id == 1)
    if status is not None:
        query = query.filter(Roster.status == status)
    rows = query.all()
    items: List[RosterModel] = []
    for roster, opp in rows:
        item = RosterModel(
            id=roster.id,
            user_id=roster.user_id,
            opportunity_id=roster.opportunity_id,
            opportunity_title=opp.title if opp is not None else None,
            role=roster.role,
            status=roster.status,
            created_at=roster.created_at,
            updated_at=roster.updated_at,
        )
        items.append(item)
    response = RostersListResponse(rosters=items)
    session.close()
    return response


@app.get(
    "/api/me/rosters/by-opportunity-title",
    response_model=RosterResponse,
    summary="Get my roster entry by opportunity title",
    description="Retrieve the roster entry for the user in an opportunity with a title.",
    tags=["rosters"],
    operation_id="get_my_roster_by_opportunity_title",
)
async def get_my_roster_by_opportunity_title(
    opportunity_title: str = Query(..., description="Exact title of the opportunity", example="Senior Center Tech Help Volunteer")
) -> RosterResponse:
    session = SessionLocal()
    row = (
        session.query(Roster, Opportunity)
        .join(Opportunity, Opportunity.id == Roster.opportunity_id)
        .filter(Roster.user_id == 1, Opportunity.title == opportunity_title)
        .first()
    )
    if row is None:
        roster = Roster(
            id=0,
            user_id=1,
            opportunity_id=0,
            role=None,
            status="inactive",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        opp = None
    else:
        roster, opp = row
    item = RosterModel(
        id=roster.id,
        user_id=roster.user_id,
        opportunity_id=roster.opportunity_id,
        opportunity_title=opp.title if opp is not None else None,
        role=roster.role,
        status=roster.status,
        created_at=roster.created_at,
        updated_at=roster.updated_at,
    )
    response = RosterResponse(roster=item)
    session.close()
    return response


@app.get(
    "/api/me/shift-assignments",
    response_model=ShiftAssignmentsListResponse,
    summary="List my shift assignments with filters",
    description="Retrieve the user's shift assignments with optional filters.",
    tags=["shift_assignments"],
    operation_id="list_my_shift_assignments",
)
async def list_my_shift_assignments(
    opportunity_id: Optional[int] = Query(None, description="Filter by associated opportunity identifier", example=42),
    opportunity_title: Optional[str] = Query(None, description="Filter by exact opportunity title", example="Park Cleanup Day – Golden Gate Park"),
    status: Optional[str] = Query(None, description="Filter by assignment status", example="assigned"),
    start_from: Optional[datetime] = Query(None, description="Include shifts with start_datetime >= this timestamp", example="2025-03-01T00:00:00Z"),
    start_to: Optional[datetime] = Query(None, description="Include shifts with start_datetime <= this timestamp", example="2025-03-31T23:59:59Z"),
) -> ShiftAssignmentsListResponse:
    session = SessionLocal()
    query = (
        session.query(ShiftAssignment, OpportunityShift, Opportunity)
        .join(OpportunityShift, OpportunityShift.id == ShiftAssignment.shift_id)
        .join(Opportunity, Opportunity.id == ShiftAssignment.opportunity_id)
        .filter(ShiftAssignment.user_id == 1)
    )
    if opportunity_id is not None:
        query = query.filter(ShiftAssignment.opportunity_id == opportunity_id)
    if opportunity_title is not None:
        query = query.filter(Opportunity.title == opportunity_title)
    if status is not None:
        query = query.filter(ShiftAssignment.status == status)
    if start_from is not None:
        query = query.filter(OpportunityShift.start_datetime >= start_from)
    if start_to is not None:
        query = query.filter(OpportunityShift.start_datetime <= start_to)
    rows = query.all()
    items: List[ShiftAssignmentModel] = []
    for assign, shift, opp in rows:
        item = ShiftAssignmentModel(
            id=assign.id,
            user_id=assign.user_id,
            opportunity_id=assign.opportunity_id,
            opportunity_title=opp.title if opp is not None else None,
            shift_id=assign.shift_id,
            shift_start_datetime=shift.start_datetime if shift is not None else None,
            shift_end_datetime=shift.end_datetime if shift is not None else None,
            status=assign.status,
            note=assign.note,
            created_at=assign.created_at,
            updated_at=assign.updated_at,
        )
        items.append(item)
    response = ShiftAssignmentsListResponse(shift_assignments=items)
    session.close()
    return response


@app.patch(
    "/api/me/shift-assignments/{assignment_id}",
    response_model=ShiftAssignmentResponse,
    summary="Update status or note on my shift assignment",
    description="Update the status and/or note on a specific shift assignment.",
    tags=["shift_assignments"],
    operation_id="update_my_shift_assignment",
)
async def update_my_shift_assignment(
    assignment_id: int = Path(..., description="Shift assignment identifier", example=400),
    body: UpdateShiftAssignmentRequest = Body(...),
) -> ShiftAssignmentResponse:
    session = SessionLocal()
    row = (
        session.query(ShiftAssignment, OpportunityShift, Opportunity)
        .join(OpportunityShift, OpportunityShift.id == ShiftAssignment.shift_id)
        .join(Opportunity, Opportunity.id == ShiftAssignment.opportunity_id)
        .filter(ShiftAssignment.id == assignment_id, ShiftAssignment.user_id == 1)
        .first()
    )
    if row is None:
        now = datetime.utcnow()
        assignment = ShiftAssignment(
            id=assignment_id,
            user_id=1,
            opportunity_id=0,
            shift_id=0,
            status=body.status if body.status is not None else "assigned",
            note=body.note,
            created_at=now,
            updated_at=now,
        )
        session.add(assignment)
        session.commit()
        session.refresh(assignment)
        shift = None
        opp = None
    else:
        assignment, shift, opp = row
        if body.status is not None:
            assignment.status = body.status
        if body.note is not None:
            assignment.note = body.note
        assignment.updated_at = datetime.utcnow()
        session.commit()
        session.refresh(assignment)
    if row is None:
        shift_start = None
        shift_end = None
        opp_title = None
    else:
        shift_start = shift.start_datetime if shift is not None else None
        shift_end = shift.end_datetime if shift is not None else None
        opp_title = opp.title if opp is not None else None
    model = ShiftAssignmentModel(
        id=assignment.id,
        user_id=assignment.user_id,
        opportunity_id=assignment.opportunity_id,
        opportunity_title=opp_title,
        shift_id=assignment.shift_id,
        shift_start_datetime=shift_start,
        shift_end_datetime=shift_end,
        status=assignment.status,
        note=assignment.note,
        created_at=assignment.created_at,
        updated_at=assignment.updated_at,
    )
    response = ShiftAssignmentResponse(shift_assignment=model)
    session.close()
    return response


@app.post(
    "/api/me/shift-assignments/{assignment_id}/notes",
    response_model=ShiftNoteResponse,
    summary="Add a note to my shift assignment",
    description="Create a new note record attached to a shift assignment.",
    tags=["shift_notes"],
    operation_id="create_shift_note_for_my_assignment",
)
async def create_shift_note_for_my_assignment(
    assignment_id: int = Path(..., description="Shift assignment identifier", example=400),
    body: CreateShiftNoteRequest = Body(...),
) -> ShiftNoteResponse:
    session = SessionLocal()
    now = datetime.utcnow()
    note = ShiftNote(
        shift_assignment_id=assignment_id,
        user_id=1,
        note=body.note,
        created_at=now,
        updated_at=now,
    )
    session.add(note)
    session.commit()
    session.refresh(note)
    response = ShiftNoteResponse(shift_note=note)
    session.close()
    return response


@app.get(
    "/api/me/shift-assignments/{assignment_id}/notes",
    response_model=ShiftNotesListResponse,
    summary="List notes on my shift assignment",
    description="Retrieve all notes created for a specific shift assignment.",
    tags=["shift_notes"],
    operation_id="list_shift_notes_for_my_assignment",
)
async def list_shift_notes_for_my_assignment(
    assignment_id: int = Path(..., description="Shift assignment identifier", example=400)
) -> ShiftNotesListResponse:
    session = SessionLocal()
    notes = (
        session.query(ShiftNote)
        .filter(ShiftNote.shift_assignment_id == assignment_id, ShiftNote.user_id == 1)
        .all()
    )
    response = ShiftNotesListResponse(notes=notes)
    session.close()
    return response


@app.post(
    "/api/me/volunteer-hours",
    response_model=VolunteerHoursResponse,
    summary="Log volunteer hours for an opportunity",
    description="Create a volunteer_hours record linked to the authenticated user.",
    tags=["volunteer_hours"],
    operation_id="create_my_volunteer_hours",
)
async def create_my_volunteer_hours(
    body: VolunteerHoursCreateRequest = Body(...)
) -> VolunteerHoursResponse:
    session = SessionLocal()
    now = datetime.utcnow()
    record = VolunteerHour(
        user_id=1,
        opportunity_id=body.opportunity_id,
        shift_id=body.shift_id,
        date=body.service_date,
        start_datetime=body.start_datetime,
        end_datetime=body.end_datetime,
        hours=body.hours,
        comment=body.comment,
        created_at=now,
        updated_at=now,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    model = VolunteerHoursModel(
        id=record.id,
        user_id=record.user_id,
        opportunity_id=record.opportunity_id,
        shift_id=record.shift_id,
        service_date=record.date,
        start_datetime=record.start_datetime,
        end_datetime=record.end_datetime,
        hours=record.hours,
        comment=record.comment,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
    response = VolunteerHoursResponse(volunteer_hours=model)
    session.close()
    return response


@app.get(
    "/api/me/volunteer-hours",
    response_model=VolunteerHoursListResponse,
    summary="List my volunteer hours",
    description="Retrieve volunteer hours entries for the user with optional date filters.",
    tags=["volunteer_hours"],
    operation_id="list_my_volunteer_hours",
)
async def list_my_volunteer_hours(
    date_from: Optional[date] = Query(None, description="Include records where date >= this", example="2025-01-01"),
    date_to: Optional[date] = Query(None, description="Include records where date <= this", example="2025-12-31"),
) -> VolunteerHoursListResponse:
    session = SessionLocal()
    query = session.query(VolunteerHour, Opportunity).join(Opportunity, Opportunity.id == VolunteerHour.opportunity_id).filter(VolunteerHour.user_id == 1)
    if date_from is not None:
        query = query.filter(VolunteerHour.date >= date_from)
    if date_to is not None:
        query = query.filter(VolunteerHour.date <= date_to)
    rows = query.all()
    items: List[VolunteerHoursListItemModel] = []
    for vh, opp in rows:
        item = VolunteerHoursListItemModel(
            id=vh.id,
            user_id=vh.user_id,
            opportunity_id=vh.opportunity_id,
            opportunity_title=opp.title if opp is not None else None,
            service_date=vh.date,
            hours=vh.hours,
            comment=vh.comment,
        )
        items.append(item)
    response = VolunteerHoursListResponse(volunteer_hours=items)
    session.close()
    return response


@app.get(
    "/api/me/reports/hours-by-opportunity-and-cause",
    response_model=HoursSummaryResponse,
    summary="Get my hours summary by opportunity and cause",
    description="Aggregate volunteer hours in a year for the user grouped by opportunity and cause.",
    tags=["reports"],
    operation_id="get_my_hours_summary_by_opportunity_and_cause",
)
async def get_my_hours_summary_by_opportunity_and_cause(
    year: int = Query(..., description="Calendar year for which to summarize hours", example=2025),
    top_opportunities_limit: Optional[int] = Query(None, description="Maximum number of top opportunities to include", example=3),
) -> HoursSummaryResponse:
    session = SessionLocal()
    start_date_year = date(year, 1, 1)
    end_date_year = date(year, 12, 31)
    rows = (
        session.query(VolunteerHour, Opportunity, Cause)
        .join(Opportunity, Opportunity.id == VolunteerHour.opportunity_id)
        .outerjoin(Cause, Cause.id == Opportunity.cause_id)
        .filter(VolunteerHour.user_id == 1)
        .filter(VolunteerHour.date >= start_date_year)
        .filter(VolunteerHour.date <= end_date_year)
        .all()
    )
    total_hours = 0.0
    by_opp: Dict[int, HoursByOpportunityItem] = {}
    by_cause: Dict[Optional[int], HoursByCauseItem] = {}
    for vh, opp, cause in rows:
        total_hours += vh.hours
        if opp.id not in by_opp:
            by_opp[opp.id] = HoursByOpportunityItem(
                opportunity_id=opp.id,
                opportunity_title=opp.title,
                cause_id=opp.cause_id,
                cause_name=cause.name if cause is not None else None,
                total_hours=0.0,
            )
        opp_item = by_opp[opp.id]
        opp_item.total_hours = opp_item.total_hours + vh.hours
        cause_id = opp.cause_id
        if cause_id not in by_cause:
            by_cause[cause_id] = HoursByCauseItem(
                cause_id=cause_id,
                cause_name=cause.name if cause is not None else None,
                total_hours=0.0,
            )
        cause_item = by_cause[cause_id]
        cause_item.total_hours = cause_item.total_hours + vh.hours
    hours_by_opportunity = list(by_opp.values())
    hours_by_cause = list(by_cause.values())
    hours_by_opportunity_sorted = sorted(hours_by_opportunity, key=lambda x: x.total_hours, reverse=True)
    if top_opportunities_limit is not None:
        top_opp_list = hours_by_opportunity_sorted[:top_opportunities_limit]
    else:
        top_opp_list = hours_by_opportunity_sorted
    top_opportunities: List[TopOpportunityItem] = []
    for item in top_opp_list:
        top_item = TopOpportunityItem(
            opportunity_id=item.opportunity_id,
            opportunity_title=item.opportunity_title,
            total_hours=item.total_hours,
        )
        top_opportunities.append(top_item)
    response = HoursSummaryResponse(
        year=year,
        total_hours=total_hours,
        hours_by_opportunity=hours_by_opportunity,
        hours_by_cause=hours_by_cause,
        top_opportunities=top_opportunities,
    )
    session.close()
    return response


@app.get(
    "/api/opportunities/{opportunity_id}/ratings",
    response_model=OpportunityRatingsSummaryResponse,
    summary="Get ratings summary for an opportunity",
    description="Retrieve average rating and count for a given opportunity.",
    tags=["opportunity_ratings"],
    operation_id="get_opportunity_ratings_summary",
)
async def get_opportunity_ratings_summary(
    opportunity_id: int = Path(..., description="Opportunity identifier", example=80)
) -> OpportunityRatingsSummaryResponse:
    session = SessionLocal()
    rows = session.query(OpportunityRating.rating).filter(OpportunityRating.opportunity_id == opportunity_id).all()
    count = len(rows)
    if count == 0:
        avg = None
    else:
        total = 0.0
        for (rating,) in rows:
            total += float(rating)
        avg = total / float(count)
    response = OpportunityRatingsSummaryResponse(
        opportunity_id=opportunity_id,
        average_rating=avg,
        ratings_count=count,
    )
    session.close()
    return response


@app.get(
    "/api/me/favorites",
    response_model=FavoritesListResponse,
    summary="List my favorite opportunities",
    description="Retrieve all opportunities saved as favorites by the user.",
    tags=["favorites"],
    operation_id="list_my_favorites",
)
async def list_my_favorites() -> FavoritesListResponse:
    session = SessionLocal()
    rows = (
        session.query(Favorite, Opportunity)
        .join(Opportunity, Opportunity.id == Favorite.opportunity_id)
        .filter(Favorite.user_id == 1)
        .all()
    )
    items: List[FavoriteModel] = []
    for fav, opp in rows:
        item = FavoriteModel(
            id=fav.id,
            user_id=fav.user_id,
            opportunity_id=fav.opportunity_id,
            opportunity_title=opp.title if opp is not None else None,
            note=fav.note,
            created_at=fav.created_at,
            updated_at=fav.updated_at,
        )
        items.append(item)
    response = FavoritesListResponse(favorites=items)
    session.close()
    return response


@app.post(
    "/api/me/favorites",
    response_model=FavoriteResponse,
    summary="Add an opportunity to my favorites",
    description="Create a favorite record linking the current user to an opportunity.",
    tags=["favorites"],
    operation_id="create_my_favorite",
)
async def create_my_favorite(
    body: CreateFavoriteRequest = Body(...)
) -> FavoriteResponse:
    session = SessionLocal()
    fav = session.query(Favorite).filter(Favorite.user_id == 1, Favorite.opportunity_id == body.opportunity_id).first()
    now = datetime.utcnow()
    if fav is None:
        fav = Favorite(
            user_id=1,
            opportunity_id=body.opportunity_id,
            note=body.note,
            created_at=now,
            updated_at=now,
        )
        session.add(fav)
        session.commit()
        session.refresh(fav)
    else:
        if body.note is not None:
            fav.note = body.note
        fav.updated_at = now
        session.commit()
        session.refresh(fav)
    opp = session.query(Opportunity).filter(Opportunity.id == fav.opportunity_id).first()
    model = FavoriteModel(
        id=fav.id,
        user_id=fav.user_id,
        opportunity_id=fav.opportunity_id,
        opportunity_title=opp.title if opp is not None else None,
        note=fav.note,
        created_at=fav.created_at,
        updated_at=fav.updated_at,
    )
    response = FavoriteResponse(favorite=model)
    session.close()
    return response


@app.patch(
    "/api/me/favorites/{favorite_id}",
    response_model=FavoriteResponse,
    summary="Update my note on a favorite opportunity",
    description="Update the note for an existing favorite record for the user.",
    tags=["favorites"],
    operation_id="update_my_favorite_note",
)
async def update_my_favorite_note(
    favorite_id: int = Path(..., description="Favorite record identifier", example=900),
    body: UpdateFavoriteRequest = Body(...),
) -> FavoriteResponse:
    session = SessionLocal()
    fav = (
        session.query(Favorite)
        .filter(Favorite.id == favorite_id, Favorite.user_id == 1)
        .first()
    )
    now = datetime.utcnow()
    if fav is None:
        fav = Favorite(
            id=favorite_id,
            user_id=1,
            opportunity_id=0,
            note=body.note,
            created_at=now,
            updated_at=now,
        )
        session.add(fav)
        session.commit()
        session.refresh(fav)
    else:
        fav.note = body.note
        fav.updated_at = now
        session.commit()
        session.refresh(fav)
    opp = session.query(Opportunity).filter(Opportunity.id == fav.opportunity_id).first()
    model = FavoriteModel(
        id=fav.id,
        user_id=fav.user_id,
        opportunity_id=fav.opportunity_id,
        opportunity_title=opp.title if opp is not None else None,
        note=fav.note,
        created_at=fav.created_at,
        updated_at=fav.updated_at,
    )
    response = FavoriteResponse(favorite=model)
    session.close()
    return response


@app.delete(
    "/api/me/favorites/{favorite_id}",
    response_model=DeleteSuccessResponse,
    summary="Remove an opportunity from my favorites",
    description="Delete a favorite record for the authenticated user.",
    tags=["favorites"],
    operation_id="delete_my_favorite",
)
async def delete_my_favorite(
    favorite_id: int = Path(..., description="Favorite record identifier to delete", example=900)
) -> DeleteSuccessResponse:
    session = SessionLocal()
    fav = (
        session.query(Favorite)
        .filter(Favorite.id == favorite_id, Favorite.user_id == 1)
        .first()
    )
    success = False
    if fav is not None:
        session.delete(fav)
        session.commit()
        success = True
    session.close()
    return DeleteSuccessResponse(success=success)


@app.get(
    "/api/opportunities/search-with-shift-filters",
    response_model=ShiftFilteredOpportunitiesResponse,
    summary="Search opportunities filtered by shift properties",
    description="Search opportunities with base filters and ensure at least one matching shift.",
    tags=["opportunities", "opportunity_shifts"],
    operation_id="search_opportunities_with_shift_filters",
)
async def search_opportunities_with_shift_filters(
    location_city: Optional[str] = Query(None, description="Filter by city", example="Chicago"),
    location_state: Optional[str] = Query(None, description="Filter by state", example="IL"),
    cause_name: Optional[str] = Query(None, description="Filter by cause name", example="animal welfare"),
    is_virtual: Optional[bool] = Query(None, description="Filter by virtual flag", example=False),
    max_hours_per_week_lte: Optional[float] = Query(None, description="Filter by max_hours_per_week <= value", example=4.0),
    required_shift_weekday: Optional[int] = Query(None, description="Require at least one shift weekday matches this", example=5),
    shift_start_from: Optional[datetime] = Query(None, description="Only consider shifts starting >= this datetime", example="2025-01-01T00:00:00Z"),
    shift_start_to: Optional[datetime] = Query(None, description="Only consider shifts starting <= this datetime", example="2025-12-31T23:59:59Z"),
    limit: Optional[int] = Query(None, description="Maximum number of opportunities", example=5),
) -> ShiftFilteredOpportunitiesResponse:
    session = SessionLocal()
    query = (
        session.query(Opportunity, Organization, OpportunityShift, Cause)
        .join(Organization, Organization.id == Opportunity.organization_id)
        .join(OpportunityShift, OpportunityShift.opportunity_id == Opportunity.id)
        .outerjoin(Cause, Cause.id == Opportunity.cause_id)
    )
    if location_city is not None:
        query = query.filter(Opportunity.location_city == location_city)
    if location_state is not None:
        query = query.filter(Opportunity.location_state == location_state)
    if cause_name is not None:
        pattern_c = f"%{cause_name}%"
        query = query.filter(Cause.name.ilike(pattern_c))
    if is_virtual is not None:
        query = query.filter(Opportunity.is_virtual == (1 if is_virtual else 0))
    if max_hours_per_week_lte is not None:
        query = query.filter(Opportunity.max_hours_per_week <= max_hours_per_week_lte)
    if shift_start_from is not None:
        query = query.filter(OpportunityShift.start_datetime >= shift_start_from)
    if shift_start_to is not None:
        query = query.filter(OpportunityShift.start_datetime <= shift_start_to)
    rows = query.all()
    best_shift_per_opp: Dict[int, OpportunityShift] = {}
    org_per_opp: Dict[int, Organization] = {}
    opp_per_id: Dict[int, Opportunity] = {}
    for opp, org, shift, cause in rows:
        if required_shift_weekday is not None:
            weekday = shift.start_datetime.weekday()
            if weekday != required_shift_weekday:
                continue
        opp_per_id[opp.id] = opp
        org_per_opp[opp.id] = org
        existing = best_shift_per_opp.get(opp.id)
        if existing is None or shift.start_datetime < existing.start_datetime:
            best_shift_per_opp[opp.id] = shift
    items: List[ShiftFilteredOpportunityModel] = []
    for oid, opp in opp_per_id.items():
        org = org_per_opp.get(oid)
        shift = best_shift_per_opp.get(oid)
        item = ShiftFilteredOpportunityModel(
            id=opp.id,
            title=opp.title,
            organization_id=opp.organization_id,
            organization_name=org.name if org is not None else None,
            start_date=opp.start_date,
            next_shift_start_datetime=shift.start_datetime if shift is not None else None,
        )
        items.append(item)
    items.sort(key=lambda x: x.next_shift_start_datetime if x.next_shift_start_datetime is not None else datetime.max)
    if limit is not None:
        items = items[:limit]
    response = ShiftFilteredOpportunitiesResponse(opportunities=items)
    session.close()
    return response


@app.get(
    "/api/me/opportunities/{opportunity_id}/last-completed-shift",
    response_model=LastCompletedShiftResponse,
    summary="Get my last completed shift in an opportunity",
    description="Retrieve the most recent completed shift assignment for the opportunity.",
    tags=["shift_assignments"],
    operation_id="get_my_last_completed_shift_for_opportunity",
)
async def get_my_last_completed_shift_for_opportunity(
    opportunity_id: int = Path(..., description="Opportunity identifier", example=80)
) -> LastCompletedShiftResponse:
    session = SessionLocal()
    row = (
        session.query(ShiftAssignment, OpportunityShift)
        .join(OpportunityShift, OpportunityShift.id == ShiftAssignment.shift_id)
        .filter(
            ShiftAssignment.user_id == 1,
            ShiftAssignment.opportunity_id == opportunity_id,
            ShiftAssignment.status == "completed",
        )
        .order_by(OpportunityShift.start_datetime.desc())
        .first()
    )
    if row is None:
        response = LastCompletedShiftResponse(shift_assignment=None)
        session.close()
        return response
    assign, shift = row
    model = LastCompletedShiftModel(
        id=assign.id,
        shift_id=assign.shift_id,
        shift_start_datetime=shift.start_datetime if shift is not None else None,
        shift_end_datetime=shift.end_datetime if shift is not None else None,
        status=assign.status,
    )
    response = LastCompletedShiftResponse(shift_assignment=model)
    session.close()
    return response


if __name__ == "__main__":
    import uvicorn, os

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    import os
    host = os.environ.get('HOST', '127.0.0.1')
    port = os.environ.get('PORT', 8001)
    print(f'Server starting on port={port}')
    from fastapi_mcp import FastApiMCP
    mcp = FastApiMCP(app)
    mcp.mount_http()
    print("MCP server enabled, please visit http://127.0.0.1:8001/mcp for the MCP service")
    uvicorn.run(app, host=host, port=int(port))