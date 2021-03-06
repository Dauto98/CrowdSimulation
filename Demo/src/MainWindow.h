#ifndef _mainwindow_h_
#define _mainwindow_h_

#include <QMainWindow>
#include <QGraphicsScene>
#include <vector>

#include "ped_model.h"
#include "ped_agent.h"
#include "ViewAgent.h"
class QGraphicsView;


class MainWindow : public QMainWindow {
public:
	MainWindow() = delete;
	MainWindow(const Ped::Model &model);

	// paint is called after each computational step
	// to repaint the window
	void paint();

	static int cellToPixel(int val);
	static const int cellsizePixel = 5;
	static const int width = 800;
	static const int height = 600;
	~MainWindow();
private:
	QGraphicsView *graphicsView;
	QGraphicsScene * scene;

	const Ped::Model &model;

	// a reference to the list of agent coordinate
	const std::vector<float>& agentsX;
	const std::vector<float>& agentsY;

	std::vector<QGraphicsRectItem *> rectList;

	// the graphical representation of each agent
	std::vector<ViewAgent*> viewAgents;

	// The pixelmap containing the heatmap image (Assignment 4)
	QGraphicsPixmapItem *pixmap;
};

#endif
