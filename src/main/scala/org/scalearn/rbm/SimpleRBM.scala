package org.scalearn.rbm

import scala.util.Random
import scala.math
import scala.util.parsing.json._

class SimpleRBM() {

  //Configurations
  var biasVisible: Layer = null
  var biasHidden: Layer = null
  var weights: Array[Layer] = null
  var gaussianVisibles: Boolean = false

  //XXHT: what is the role of scale
  protected val scale: Float = 0.001f
  private var distribution: ProbabilityDistribution = null

  def initializeConfiguration(numVisible: Int, numHidden: Int, gVisibles: Boolean, generator: Random = null): Unit = {
    val rand: Random = if (generator == null) new Random() else generator
    distribution = ProbabilityDistribution(rand)
    biasVisible = Layer(numVisible)
    biasHidden = Layer(numHidden)
    weights = new Array(numHidden)
    gaussianVisibles = gVisibles

    //initialize vectors
    for (i <- 0 until numVisible)
      biasVisible.set(i, scale * rand.nextGaussian().toFloat)
    for (i <- 0 until numHidden)
      biasHidden.set(i, scale * rand.nextGaussian().toFloat)

    for (i <- 0 until numHidden) {
      weights(i) = Layer(numVisible)
      for (j <- 0 until numVisible) {
        weights(i).set(j, 2 * scale * rand.nextGaussian().toFloat)
      }
    }
  }

  //Given visible, activate hidden layer
  def activateHidden(visible: Layer, bias: Layer): Layer = {

    var h: Layer = Layer(biasHidden.size);

    if (visible.size != biasVisible.size)
      throw new Exception("Number of visible nodes do not match bias nodes: visible= " + visible.size + " biasVisible=" + biasVisible.size)

    if (bias != null && h.size != bias.size && bias.size > 1)
      throw new AssertionError("Hidden nodes do not match bias nodes")

    //Activate hidden nodes
    for (i <- 0 until weights.length; k <- 0 until visible.size)
      h.+(i, weights(i)(k) * visible(k))

    //Add bias nodes
    for (i <- 0 until h.size) {
      var inputBias: Float = 0.0f
      //input bias
      if (bias != null && bias.size != 0)
        inputBias = if (bias.size == 1) bias(0) else bias(i)

      h.set(i, Utilities.sigmoid(h(i) + biasHidden(i) + inputBias))
    }

    h
  }

  // Given hidden configuration, generate visible nodes
  def activateVisible(hidden: Layer, bias: Layer): Layer = {
    var v: Layer = Layer(biasVisible.size)

    if (bias != null && v.size != bias.size && bias.size > 1)
      throw new AssertionError("bias must be 0,1 or visible length");

    //Update visible 
    for (k <- 0 until weights.length; i <- 0 until v.size)
      v.+(i, weights(k)(i) * hidden(k))

    //Add visible bias
    for (i <- 0 until v.size) {
      v.+(i, biasVisible(i));

      //Add input bias
      if (bias != null && bias.size != 0)
        v.+(i, if (bias.size == 1) bias(0) else bias(i))

      if (!gaussianVisibles)
        v.set(i, Utilities.sigmoid(v(i)))
    }

    v
  }

  //Free energy of the configuration
  //TODO: is this formula correct?
  def freeEnergy(): Float = {
    var energy: Float = 0.0f
    for (j <- 0 until biasHidden.size; i <- 0 until biasVisible.size)
      energy -= biasVisible(i) * biasHidden(j) * weights(j)(i)

    energy
  }

  def forwardGibbsSampler(input: Layer): Iterator[Tuple] = {
    return gibbsSamplerHVH(input, TupleFactory(input))
  }

  def reverseGibbsSampler(input: Layer): Iterator[Tuple] = {
    return gibbsSamplerVHV(input, TupleFactory(input))
  }

  def gibbsSamplerHVH(visible: Layer, tfactory: TupleFactory): Iterator[Tuple] = new Iterator[Tuple]() {
    var v: Layer = visible
    var h: Layer = activateHidden(v, null)

    override def hasNext(): Boolean = return true

    override def next(): Tuple = {
      val t: Tuple = tfactory.create(v, h)
      //TODO: is this correct gibbs_hvh?
      v = activateVisible(distribution.bernoulli(h), null)
      h = activateHidden(v, null)

      t
    }
  }

  def gibbsSamplerVHV(hidden: Layer, tfactory: TupleFactory): Iterator[Tuple] = new Iterator[Tuple]() {
    var v: Layer = activateVisible(distribution.bernoulli(hidden), null)
    var h: Layer = hidden

    override def hasNext(): Boolean = return true

    override def next(): Tuple = {
      val t: Tuple = tfactory.create(v, h)

      //is this correct gibbs_vhv?
      v = activateVisible(distribution.bernoulli(h), null);
      h = activateHidden(v, null);
      t
    }
  }
}

object SimpleRBM {
  def apply(numVisible: Int, numHidden: Int, gaussianVisibles: Boolean, generator: Random = null): SimpleRBM = {
    val model: SimpleRBM = new SimpleRBM()
    model.initializeConfiguration(numVisible, numHidden, gaussianVisibles, generator)
    model
  }

  def apply(bVisible: Array[Float], bHidden: Array[Float], w: Array[Array[Float]], gVisibles: Boolean): SimpleRBM = {
    val model: SimpleRBM = new SimpleRBM()
    model.biasVisible = Layer(bVisible)
    model.biasHidden = Layer(bHidden)
    model.weights = for (l <- w) yield Layer(l)
    model.gaussianVisibles = gVisibles
    model
  }

  //TODO: json is too big. Need to serialize data.
  def toJSON(model: SimpleRBM): JSONObject = {

    val links = for (w <- model.weights) yield new JSONArray(w.toArray.toList)

    new JSONObject(Map("gaussianVisibles" -> model.gaussianVisibles,
      "biasVisible" -> new JSONArray(model.biasVisible.toArray.toList),
      "biasHidden" -> new JSONArray(model.biasHidden.toArray.toList),
      "weights" -> new JSONArray(links.toList)))
  }

  def apply(model: Map[String, Any]): SimpleRBM = {
    val gaussianVisibles = model("gaussianVisibles").asInstanceOf[Boolean]
    val biasVisible: Array[Float] = model("biasVisible").asInstanceOf[Array[Float]]
    val biasHidden: Array[Float] = model("biasHidden").asInstanceOf[Array[Float]]
    val weights: Array[Array[Float]] = model("weights").asInstanceOf[Array[Array[Float]]]
    SimpleRBM(biasVisible, biasHidden, weights, gaussianVisibles)
  }

}
  

