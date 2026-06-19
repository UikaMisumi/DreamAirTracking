using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class PupilAssistEstimatorTests
{
    [Fact]
    public void WideEyesConstrictAssistedPupil()
    {
        var normal = PupilAssistEstimator.Estimate(0.92, 0.8);
        var wide = PupilAssistEstimator.Estimate(1.0, 0.8);

        Assert.True(normal.Found);
        Assert.True(wide.Found);
        Assert.True(wide.Normalized < normal.Normalized);
        Assert.InRange(normal.Normalized, 0.49, 0.51);
        Assert.InRange(wide.Normalized, 0.19, 0.21);
    }

    [Fact]
    public void LinkedAssistUsesEitherEyeAndKeepsBothPupilsTogether()
    {
        var (left, right) = PupilAssistEstimator.EstimateLinked(
            leftOpenness: 1.0,
            leftConfidence: 0.8,
            rightOpenness: 0.62,
            rightConfidence: 0.8);

        Assert.True(left.Found);
        Assert.True(right.Found);
        Assert.Equal(left.Normalized, right.Normalized);
        Assert.InRange(left.Normalized, 0.19, 0.21);
    }

    [Fact]
    public void LinkedAssistStaysNeutralUntilEitherEyeIsClearlyWide()
    {
        var (left, right) = PupilAssistEstimator.EstimateLinked(
            leftOpenness: 0.92,
            leftConfidence: 0.8,
            rightOpenness: 0.90,
            rightConfidence: 0.8);

        Assert.True(left.Found);
        Assert.True(right.Found);
        Assert.Equal(PupilAssistEstimator.NeutralNormalized, left.Normalized);
        Assert.Equal(left.Normalized, right.Normalized);
    }

    [Fact]
    public void ApertureAssistStaysNeutralForNormalOpenEye()
    {
        var pupil = PupilAssistEstimator.EstimateFromAperture(1.0, 0.8, 74);

        Assert.True(pupil.Found);
        Assert.Equal(PupilAssistEstimator.NeutralNormalized, pupil.Normalized);
    }

    [Fact]
    public void ApertureAssistConstrictsOnlyWhenClearlyWide()
    {
        var pupil = PupilAssistEstimator.EstimateFromAperture(1.0, 0.8, 86);

        Assert.True(pupil.Found);
        Assert.InRange(pupil.Normalized, 0.19, 0.21);
    }

    [Fact]
    public void ApertureAssistHasGentleMidWideResponse()
    {
        var pupil = PupilAssistEstimator.EstimateFromAperture(1.0, 0.8, 80);

        Assert.True(pupil.Found);
        Assert.InRange(pupil.Normalized, 0.28, 0.45);
    }

    [Theory]
    [InlineData(0.2, 0.8)]
    [InlineData(0.9, 0.01)]
    public void GatedInputDoesNotPublishPupil(double openness, double confidence)
    {
        var pupil = PupilAssistEstimator.Estimate(openness, confidence);

        Assert.False(pupil.Found);
        Assert.False(pupil.Quality);
    }
}
