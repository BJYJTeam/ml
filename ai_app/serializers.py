from rest_framework import serializers


MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 5000
MAX_COMMENT_LENGTH = 4000


class MedicalQuestionSerializer(serializers.Serializer):
    post_id = serializers.CharField(max_length=128)
    title = serializers.CharField(max_length=MAX_TITLE_LENGTH, allow_blank=True)
    content = serializers.CharField(max_length=MAX_CONTENT_LENGTH, allow_blank=False)

    def validate_content(self, value):
        if not value.strip():
            raise serializers.ValidationError("This field must not be blank.")
        return value.strip()

    def validate_title(self, value):
        return value.strip()


class DoctorCommentSerializer(serializers.Serializer):
    comment_author = serializers.CharField(max_length=128, allow_blank=True)
    comment_content = serializers.CharField(max_length=MAX_COMMENT_LENGTH, allow_blank=False)

    def validate_comment_content(self, value):
        if not value.strip():
            raise serializers.ValidationError("This field must not be blank.")
        return value.strip()


class DoctorDraftSerializer(MedicalQuestionSerializer):
    comments = DoctorCommentSerializer(many=True)


class ImageRecommendationSerializer(serializers.Serializer):
    content = serializers.CharField(max_length=MAX_CONTENT_LENGTH, allow_blank=False)

    def validate_content(self, value):
        if not value.strip():
            raise serializers.ValidationError("This field must not be blank.")
        return value.strip()
