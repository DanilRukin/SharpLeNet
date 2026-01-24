using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensor = SharpLeNet.Core.Tensor;

namespace UnitTests.SharpLeNet.Core.Tests;

public class TensorTests
{
    [Fact]
    public void ShouldGetCorrectShape()
    {
        Tensor tensor = new(new double[]{ 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        tensor.Shape.Should().BeSameAs(new int[] { 2, 3 });
    }
}
