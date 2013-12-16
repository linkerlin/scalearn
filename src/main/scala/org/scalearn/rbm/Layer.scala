package org.scalearn.rbm

import java.awt.image.{ BufferedImage }
import java.io.DataInput
import java.io.DataOutput

trait Layer {
  def apply(i: Int): Float
  def length: Int
  def toArray: Array[Float]
  final def size = length
  def set(i: Int, v: Float)
  def +(i: Int, incr: Float): Unit
  def *(i: Int, incr: Float): Unit
  def /(i: Int, incr: Float): Unit
  def zero(): Unit
  //TODO:temporary
  def display(msg: String): Unit
  def foreachElement(f: (Int, Double) => Unit): Unit = { val l = length; var i = 0; while (i < l) { f(i, apply(i)); i += 1 } }
  def toBinary: Unit
  def toGaussian: Unit
  def mean: Float
  def stddev: Float
  def save(output:DataOutput) = {
    output.write(Layer.MAGIC)
    output.writeInt(size)
    for(i <- 0 until size){
      output.writeFloat(this(i))
    }
  }
  
}
class MutableLayer(array: Array[Float]) extends Layer {
  private val a = array
  private lazy val m: Float = Utilities.mean(this)
  private lazy val s: Float = Utilities.stddev(this, m)
  def mean: Float = m
  def stddev: Float = s
  def length = a.length
  def apply(i: Int) = a(i)
  def +(i: Int, v: Float) = a(i) += v
  def *(i: Int, v: Float) = a(i) *= v
  def /(i: Int, v: Float) = a(i) /= v
  def zero(): Unit = for (i <- 0 until a.size) a(i) = 0.0f
  def set(i: Int, v: Float) = a(i) = v
  def toArray: Array[Float] = a
  def display(msg: String): Unit = {
    print("Instance:" + msg + " ")
    a.foreach(x => print(x + " "))
    println()
  }

  def toGaussian: Unit = {
    val m = mean
    var d = stddev
    d = if (d < 0.1f) 0.1f else d
    for (i <- 0 until size)
      set(i, (a(i) - m) / s)

  }
  def toBinary = {
    for (i <- 0 until size)
      set(i, if (a(i) > 30) 1.0f else 0.0f)
  }
}

object Layer {
  def MAGIC = Array[Byte](7, 13,19,15)
  
  def apply(input:DataInput):Layer = {
    val magic:Array[Byte] = new Array(4)
    input.readFully(magic)
    val size = input.readInt
    val vect:Array[Float] = (for(i <- 0 until size) yield input.readFloat).toArray
    Layer(vect)
  }
  
  def apply(array: Array[Float]): Layer = new MutableLayer(array)
  def apply(size: Int): Layer = Layer(new Array[Float](size))
  def apply(from: Array[Int]): Layer = {
    val result: Layer = Layer(from.length)
    for (i <- 0 until from.length)
      result.set(i, from(i))
    result
  }
  def fromBinary(l: Layer): Array[Float] = {
    for (x <- l.toArray) yield x * 255.0f
  }

  def of(img: BufferedImage): Layer = {
    val size = img.getWidth() * img.getHeight()
    var layer: Layer = Layer(size)
    var width = 0
    var height = 0
    for (i <- 0 until size) {
      layer.set(i, img.getData().getSample(width, height, 0))
      width += 1
      if (width >= img.getWidth()) {
        width = 0
        height += 1
      }
    }

    layer
  }
  def gaussianOf(img: BufferedImage): Layer = {
    val l = of(img)
    l.toGaussian
    l
  }

}