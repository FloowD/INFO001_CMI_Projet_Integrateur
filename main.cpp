#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

struct ColorDistribution {
  float data[ 8 ][ 8 ][ 8 ]; // l'histogramme
  int nb;                     // le nombre d'échantillons
    
  ColorDistribution() { reset(); }
  ColorDistribution& operator=( const ColorDistribution& other ) = default;
  // Met à zéro l'histogramme    
  void reset(){
    for( int i = 0; i < 8; i++ )
      for( int j = 0; j < 8; j++ )
        for( int k = 0; k < 8; k++ )
          data[ i ][ j ][ k ] = 0;
    nb = 0;
  }
  // Ajoute l'échantillon color à l'histogramme:
  // met +1 dans la bonne case de l'histogramme et augmente le nb d'échantillons
  void add( Vec3b color ){
    int r = color[ 0 ] / 32;
    int g = color[ 1 ] / 32;
    int b = color[ 2 ] / 32;
    data[ r ][ g ][ b ]++;
    nb++;
  }
  // Indique qu'on a fini de mettre les échantillons:
  // divise chaque valeur du tableau par le nombre d'échantillons
  // pour que case représente la proportion des picels qui ont cette couleur.
  void finished(){
    for( int i = 0; i < 8; i++ )
      for( int j = 0; j < 8; j++ )
        for( int k = 0; k < 8; k++ )
          data[ i ][ j ][ k ] /= nb;
  }
  // Retourne la distance entre cet histogramme et l'histogramme other
  float distance( const ColorDistribution& other ) const
  {
    float dist = 0.0;
    for( int i = 0; i < 8; i++ )
      for( int j = 0; j < 8; j++ )
        for( int k = 0; k < 8; k++ ){
           float numerateur = pow(data[ i ][ j ][ k ] - other.data[ i ][ j ][ k ],2);
           float denominateur = data[ i ][ j ][ k ] + other.data[ i ][ j ][ k ];
           if ( denominateur != 0 )
            dist += numerateur / denominateur;
        }
    return dist;
  }
};


ColorDistribution
getColorDistribution( Mat input, Point pt1, Point pt2 )
{
  ColorDistribution cd;
  for ( int y = pt1.y; y < pt2.y; y++ )
    for ( int x = pt1.x; x < pt2.x; x++ )
      cd.add( input.at<Vec3b>( y, x ) );
  cd.finished();
  return cd;
}

//Retourne la plus petite distance entre h et les histogrammes de couleurs de hists
float minDistance( const ColorDistribution& h,
                   const std::vector< ColorDistribution >& hists )
{
    float min_dist = h.distance( hists[0] );
    for (int i = 1; i < hists.size(); i++)
    {
        float dist = h.distance( hists[i] );
        if ( dist < min_dist )
            min_dist = dist;
    }
    return min_dist;
}

Mat recoObject( Mat input,
                const std::vector< std::vector< ColorDistribution > >& all_col_hists,
                const std::vector< Vec3b >& colors,
                const int bloc )
{
  cv::Mat output = input.clone();
  // On parcours les blocs dans l'image
  for ( int y = 0; y < input.rows; y += bloc )
    for ( int x = 0; x < input.cols; x += bloc ){
      // On calcul l'histogramme du bloc
      Point pt1( x, y );
      Point pt2( x + bloc, y + bloc );
      ColorDistribution h = getColorDistribution( input, pt1, pt2 );
      // On calcul la distance entre l'histogramme du bloc et les histogrammes de l'objet
      //On prend la distance minimal
      float min_dist = minDistance( h, all_col_hists[0] );
      //On fait une recherche de distance minimal par rapport a tous les histogrammes (afin de trouver a quelle objet le bloc appartient)
      int indice_min_dist = 0;
      for (int i = 1; i < all_col_hists.size(); i++) {
        float distanceTemp = minDistance( h, all_col_hists[i] );
        if( distanceTemp < min_dist){
          min_dist = distanceTemp;
          indice_min_dist = i;
        }
      }

      //On prend le minimum des distances
      // //On colorie tous les pixels du bloc
      for ( int i = 0; i < bloc; i++ )
        for ( int j = 0; j < bloc; j++ )
          output.at<Vec3b>( y +i , x+j) = colors[indice_min_dist];
    }
  
  return output;
  
}
//Fonction qui nous génère autant de couleurs aléatoire qu'on veut (en fonction du nombre d'objets dans l'utilisation de la fonction recoObject)
std::vector< Vec3b > generateColors( int nb_colors )
{
  std::vector< Vec3b > colors;
  colors.push_back( Vec3b( 0, 0, 0 ) );
  for ( int i = 0; i < nb_colors-1; i++ )
    colors.push_back( Vec3b( rand() % 256, rand() % 256, rand() % 256 ) );
  return colors;
}


int main( int argc, char** argv )
{
  Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;
  VideoCapture* pCap = nullptr;
  const int width = 640;
  const int height= 480;
  const int size  = 50;
  // Ouvre la camera
  pCap = new VideoCapture( 0 );
  if( ! pCap->isOpened() ) {
    cout << "Couldn't open image / camera ";
    return 1;
  }
  // Force une camera 640x480 (pas trop grande).
  pCap->set( CAP_PROP_FRAME_WIDTH, 640 );
  pCap->set( CAP_PROP_FRAME_HEIGHT, 480 );
  (*pCap) >> img_input;
  if( img_input.empty() ) return 1; // probleme avec la camera

  Point pt1_left(0, 0);
  Point pt2_left(width / 2, height);
  Point pt1_right(width / 2, 0);
  Point pt2_right(width, height);
  
  Point pt1( width/2-size/2, height/2-size/2 );
  Point pt2( width/2+size/2, height/2+size/2 );

  std::vector<ColorDistribution> col_hists; // histogrammes du fond
  std::vector<ColorDistribution> col_hists_object; // histogrammes de l'objet
  std::vector<std::vector<ColorDistribution>> all_col_hists; // histogrammes de couleurs


  namedWindow( "input", 1 );
  imshow( "input", img_input );
  bool freeze = false;
  Mat output = img_input;
  bool reco = false;
  std::vector<Vec3b> colors;
  while ( true )
    {
      char c = (char)waitKey(50); // attend 50ms -> 20 images/s
      if ( pCap != nullptr && ! freeze )
        (*pCap) >> img_input;     // récupère l'image de la caméra
      if ( c == 27 || c == 'q' )  // permet de quitter l'application
        break;
      if ( c == 'f' ) // permet de geler l'image
        freeze = ! freeze;
      if ( c == 'v')
      {
        ColorDistribution gauche = getColorDistribution( img_input, pt1_left, pt2_left );
        ColorDistribution droite = getColorDistribution( img_input, pt1_right, pt2_right );

        float dist = gauche.distance( droite );
        cout <<"distance : " << dist << endl;
      }
      if (c == 'b')
      {
        const int bbloc = 128;
        for(int y=0; y<= height-bbloc; y += bbloc){
            for(int x=0; x <= width-bbloc; x += bbloc){
                Point pt1(x, y);
                Point pt2(x+bbloc, y+bbloc);
                ColorDistribution bloc = getColorDistribution( img_input, pt1, pt2 );
                //Mémorise dans col_hist
                col_hists.push_back( bloc );
            }
        }
        int nb_hists_background = col_hists.size();
        cout << "nb_hists_background : " << nb_hists_background << endl;
      }

      if (c == 'a')
      {
        ColorDistribution obj = getColorDistribution( img_input, pt1, pt2 );
        col_hists_object.push_back( obj );
        int nb_hists_object = col_hists_object.size();
        cout << "nb_hists_object : " << nb_hists_object << endl;
      }
      if(c == 'o'){
        all_col_hists.push_back(col_hists_object);
        col_hists_object.clear();
        cout << "On change d'objet " << endl;
      }

      if (c == 'r')
      {
        if (!reco){
          //Nous enregistrons tous les différents histogrammes
          all_col_hists.push_back(col_hists_object);
          all_col_hists.insert(all_col_hists.begin(), col_hists);
          colors = generateColors( all_col_hists.size() );
        }
        reco =! reco;
      }
      if (reco)
      {
        Mat gray;
        cvtColor(img_input, gray, COLOR_BGR2GRAY);
        Mat reco = recoObject( img_input, all_col_hists, colors, 8 );
        cvtColor(gray, img_input, COLOR_GRAY2BGR);
        output = 0.5 * reco + 0.5 * img_input; // mélange reco + caméra
      }else{
        cv::rectangle( img_input, pt1, pt2, Scalar( { 255.0, 255.0, 255.0 } ), 1 );
      }
      imshow( "input", output ); // affiche le flux video
    }
  return 0;
}